import torch
import torch.nn as nn
import torchvision
from models.visual_stream import VisualStream
#기존의 수정 전 saliency 적용 코드(중복 공간 어텐션 존재.)



class VAANet_sal_bias(VisualStream):
    def __init__(self,
                 snippet_duration=16,
                 sample_size=112,
                 n_classes=8,
                 seq_len=10,
                 pretrained_resnet101_path='',
                 audio_embed_size=256,
                 audio_n_segments=16,
                 audio_mean=0.0,
                 audio_std=1.0,
                 saliency_level='input'):
        super(VAANet_sal_bias, self).__init__(
            snippet_duration=snippet_duration,
            sample_size=sample_size,
            n_classes=n_classes,
            seq_len=seq_len,
            pretrained_resnet101_path=pretrained_resnet101_path
        )

        # self.register_buffer('audio_mean', torch.tensor(audio_mean))
        # self.register_buffer('audio_std', torch.tensor(audio_std))

        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.saliency_level = saliency_level

        a_resnet = torchvision.models.resnet18(pretrained=True)
        a_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        a_avgpool = nn.AvgPool2d(kernel_size=[8, 2])
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        self.a_fc = nn.Sequential(
            nn.Linear(a_resnet.fc.in_features, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )

        self.aa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.audio_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.audio_n_segments, self.audio_n_segments, bias=True),
            'relu': nn.ReLU(),
        })

        self.av_fc = nn.Linear(self.audio_embed_size + self.hp['k'], self.n_classes)

        self.sal_beta = nn.Parameter(torch.tensor(0.2)) #고정값 아님
        self.sal_beta_raw = nn.Parameter(torch.tensor(-1.5))  # softplus(-1.5) ≈ 0.2
    def _build_saliency_bias(self, saliency_map, seq_len, batch, target_h, target_w, device, dtype):
        """
        saliency_map:
            [B, Seq, 1, D, H, W] or [Seq, B, 1, D, H, W]
        return:
            bias: [Seq*B, m]   where m = target_h * target_w
        """
        if saliency_map.size(0) == batch and saliency_map.size(1) == seq_len:
            S = saliency_map.transpose(0, 1).contiguous()   # [Seq,B,1,D,H,W]
        elif saliency_map.size(0) == seq_len and saliency_map.size(1) == batch:
            S = saliency_map.contiguous()                   # already [Seq,B,1,D,H,W]
        else:
            raise ValueError(f"Unexpected saliency_map shape: {saliency_map.shape}")

        # [Seq*B, 1, D, H, W]
        S = S.view(seq_len * batch, 1, S.size(3), S.size(4), S.size(5))
        S = S.to(device=device, dtype=dtype).clamp_min(0) # 음수 제거

        # snippet 내부 프레임들을 평균내서 2D prior로 만듦 -> 각 snippet에 대해 하나의 saliency map만 남긴다.
        S = S.mean(dim=2, keepdim=True)   # [N,1,1,H,W] 

        # feature map spatial size에 맞춤
        S = nn.functional.adaptive_avg_pool3d(S, (1, target_h, target_w))  # [N,1,1,h,w]

        # [N, m] 2D 맵을 1D 벡터로 펼친다.
        S = S.squeeze(2).flatten(1)

        # 합이 1이 되도록 정규화: 확률분포처럼 정규화 -> 절대적인 saliency 크기가 아닌, 공간 위치들 사이의 상대적 중요도
        S = S / (S.sum(dim=1, keepdim=True) + 1e-6)

        # 정규화된 saliency 분포를 로그로 바꾼다. log-prior
        B = torch.log(S + 1e-6)

        # 샘플별 constant shift 제거 (softmax에는 영향 거의 없음, 수치 안정성용)
        B = B - B.mean(dim=1, keepdim=True)

        # 너무 큰 값 제한
        B = B.clamp(-4.0, 4.0)

        return B

    def forward(self, visual: torch.Tensor, audio: torch.Tensor, saliency_map: torch.Tensor):
        # print(f"🔍 saliency_map.shape = {saliency_map.shape}")
        # print(f"🔍 visual.shape = {visual.shape}")
        # print(f"🔍 audio.shape = {audio.shape}")

        if not hasattr(self, "_printed_input_layout"):
            print("\n[INPUT LAYOUT CHECK]")
            print("visual.shape:", tuple(visual.shape))
            print("saliency_map.shape:", tuple(saliency_map.shape))
            print("expected visual raw layout: [B, Seq, C, D, H, W]")
            print("visual first two dims -> B, Seq:", visual.size(0), visual.size(1))
            print("saliency first two dims:", saliency_map.size(0), saliency_map.size(1))
            self._printed_input_layout = True

        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        if not hasattr(self, "_printed_v2"):
            print("[VAANet] visual after norm min/max:", visual.min().item(), visual.max().item())
            self._printed_v2 = True

        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        # print(f"[INPUT] visual (Seq,B,C,D,H,W): {visual.shape}")
        # print(f"[INPUT] saliency_map (B,Seq,1,D,H,W): {saliency_map.shape}")
    
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        with torch.no_grad():
             # ResNet을 통과시켜 피처맵 생성
            F = self.resnet(visual)

            
            # saliency bias는 여기서 만듦
            saliency_bias = self._build_saliency_bias(
                saliency_map=saliency_map,
                seq_len=seq_len,
                batch=batch,
                target_h=F.size(3),
                target_w=F.size(4),
                device=F.device,
                dtype=F.dtype
            )
            
            F = torch.squeeze(F, dim=2)
            F = torch.flatten(F, start_dim=2)
            # print("F after squeeze/flatten:", F.shape)  # [384, C, T'*H'*W']
        F = self.conv0(F)  # [B x 512 x 16]

        Hs = self.sa_net['conv'](F)
        Hs = torch.squeeze(Hs, dim=1)
        Hs = self.sa_net['fc'](Hs)
        #### saliency bias 더하기
         # 여기에 saliency prior 추가
        Hs = Hs + self.sal_beta * saliency_bias
        
        As = self.sa_net['softmax'](Hs)
        As = torch.mul(As, self.hp['m'])
        alpha = As.view(seq_len, batch, self.hp['m'])

        fS = torch.mul(F, torch.unsqueeze(As, dim=1).repeat(1, self.hp['k'], 1))

        G = fS.transpose(1, 2).contiguous()
        Hc = self.cwa_net['conv'](G)
        Hc = torch.squeeze(Hc, dim=1)
        Hc = self.cwa_net['fc'](Hc)
        Ac = self.cwa_net['softmax'](Hc)
        Ac = torch.mul(Ac, self.hp['k'])
        beta = Ac.view(seq_len, batch, self.hp['k'])

        fSC = torch.mul(fS, torch.unsqueeze(Ac, dim=2).repeat(1, 1, self.hp['m']))
        fSC = torch.mean(fSC, dim=2)
        fSC = fSC.view(seq_len, batch, self.hp['k']).contiguous()
        fSC = fSC.permute(1, 2, 0).contiguous()

        # print(f"[TA] fSC: {tuple(fSC.shape)} (expect [B, K, Seq])")
        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        # print(f"[TA] Ht before fc: {tuple(Ht.shape)}, fc.in={self.ta_net['fc'].in_features}")
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = torch.mul(fSC, torch.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        fSCT = torch.mean(fSCT, dim=2)  # [bs x 512]

        # Audio branch
        # print("\n--- [Audio Debug] Before Normalization ---")
        # print(f"audio.device: {audio.device}")
        # print(f"audio_mean.device: {self.audio_mean.device}")
        # print(f"audio_std.device: {self.audio_std.device}")
        # print(f"audio.dtype: {audio.dtype}, audio_mean dtype: {self.audio_mean.dtype}")
        # print(f"audio shape: {audio.shape}, min: {audio.min().item()}, max: {audio.max().item()}")

        # print("\n--- Inside forward pass ---")
        # print(f"Before Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")
        #오디오 정규화
        # audio = (audio - self.audio_mean) / self.audio_std
        # print("audio shape: ", audio.shape)

        # print(f"After Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")

        
        # print(f"After Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")
        bs = audio.size(0)
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(self.audio_n_segments, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        audio = torch.unsqueeze(audio, dim=1)
        audio = self.a_resnet(audio)
        audio = torch.flatten(audio, start_dim=1).contiguous()
        audio = self.a_fc(audio)
        audio = audio.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        audio = audio.permute(1, 2, 0).contiguous()

        Ha = self.aa_net['conv'](audio)
        Ha = torch.squeeze(Ha, dim=1)
        Ha = self.aa_net['fc'](Ha)
        Aa = self.aa_net['relu'](Ha)

        fA = torch.mul(audio, torch.unsqueeze(Aa, dim=1).repeat(1, self.audio_embed_size, 1))
        fA = torch.mean(fA, dim=2)  # [bs x 256]

        # Fusion
        fSCTA = torch.cat([fSCT, fA], dim=1)
        output = self.av_fc(fSCTA)

        return output, alpha, beta, gamma
