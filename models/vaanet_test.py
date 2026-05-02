import torch
import torch.nn as nn
import torchvision
from models.visual_stream import VisualStream


class VAANet_test(VisualStream):
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
        super(VAANet_test, self).__init__(
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

    def forward(self, visual: torch.Tensor, audio: torch.Tensor, saliency_map: torch.Tensor=None):
        # if visual.dim() == 5:
        #     visual = visual.unsqueeze(1)
        # print("visual!!: ", visual.shape)
        # print("audio!!: ", audio.shape)

        # if visual.size(1) == 1 and not self.training:
        #     print("실행!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     # 1. Visual 파트: 기존처럼 seq_len 만큼 복제하여 시퀀스 생성
        #     bs_mock = 12
        #     visual = visual.repeat(bs_mock, 12, 1, 1, 1, 1)


        #     # 초기 audio 모양: [1, 32]
        
        #     # 2-1. 가상의 시간 축(4096) 생성
        #     # [1, 32] -> [1, 1, 32] (시간 축 추가) -> [1, 4096, 32] (시간 축 복제)
        #     audio_sequence = audio.unsqueeze(0).repeat(12, 1, 1)
    
        #     # 2-2. 가상의 배치 축(32) 생성
        #     # [1, 4096, 32] -> [32, 4096, 32] (배치 축 복제)
            
        #     audio = audio_sequence.repeat(1, 1, 1)
        # print("visual!! (복제 후): ", visual.shape) # 이제 [32, 4096, 32]가 출력됩니다.
        
        # print("audio!! (복제 후): ", audio.shape) # 이제 [32, 4096, 32]가 출력됩니다.
        

        
        # visual = visual.transpose(0, 1).contiguous()
        expected_L = None
        if hasattr(self, "ta_net") and isinstance(self.ta_net, nn.ModuleDict) and "fc" in self.ta_net:
            expected_L = int(self.ta_net["fc"].in_features)
        if expected_L is None and hasattr(self, "seq_len"):
            expected_L = int(self.seq_len)
        if expected_L is None and hasattr(self, "hp") and "L" in self.hp:
            expected_L = int(self.hp["L"])
    
        # visual: [?, ?, C, T, H, W]  -> 우리가 쓰는 표준: [seq_len, batch, C, T, H, W]
        if visual.dim() == 6 and expected_L is not None:
            # 검증 경로: [batch, seq_len, ...] 로 들어오는 경우가 있음
            if visual.size(0) != expected_L and visual.size(1) == expected_L:
                visual = visual.transpose(0, 1).contiguous()
        
    
        visual = visual.contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)
    
        # 이제 안전하게 해석
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        # print("seq len: ", seq_len)
        # print("batch: ", batch)
        # print("visual: ", visual.shape)
        
        # visual = visual.contiguous()
        # visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        # print("seq len: ", seq_len)
        # print("batch: ", batch)
        # print("visual: ", visual.shape)
        
        if saliency_map is not None and self.saliency_level == 'input':
            # 기대: saliency_map [B, Seq, 1, T, H, W] 또는 [Seq, B, 1, T, H, W]
            if saliency_map.dim() == 6 and saliency_map.size(0) == batch and saliency_map.size(1) == seq_len:
                saliency_map = saliency_map.transpose(0,1).contiguous()  # -> [Seq,B,1,T,H,W]
            # [Seq,B,1,T,H,W] -> [Seq,B,C,T,H,W]
            S = saliency_map.to(visual.device, dtype=visual.dtype).expand_as(visual)
            visual = 0.5 * visual + 0.5 * (visual * S)  # slot masking 예시
        
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        # print("visual(2): ", visual.shape)
        # with torch.no_grad():
        F = self.resnet(visual)
        # print("F(resnet output): ", F.shape)
        F = torch.squeeze(F, dim=2)
        # print("F(sqeeze): ", F.shape)
        F = torch.flatten(F, start_dim=2)
        # print("F(flatten): ", F.shape)
        F = self.conv0(F)  # [B x 512 x 16]
        # print("F(conv): ", F.shape)
        # F = F.repeat(12, 1, 1)
        # print("F(repeat): ", F.shape)

        Hs = self.sa_net['conv'](F)
        Hs = torch.squeeze(Hs, dim=1)
        Hs = self.sa_net['fc'](Hs)
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

        # 🔧 추가: temporal 길이를 FC가 기대하는 길이로 적응
        expected_L = int(self.ta_net['fc'].in_features)  # 보통 12
        if fSC.size(2) != expected_L:
            fSC = torch.nn.functional.adaptive_avg_pool1d(fSC, output_size=expected_L)

        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        # 🔧 길이 적응: FC가 기대하는 in_features로 맞추기
        # expected_L = int(self.ta_net['fc'].in_features)
        # if Ht.size(1) != expected_L:
        #     # [batch, seq_len] -> [batch, 1, seq_len] -> adaptive pool -> [batch, expected_L]
        #     Ht = torch.nn.functional.adaptive_avg_pool1d(Ht.unsqueeze(1), output_size=expected_L).squeeze(1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        # gamma = At.view(batch, seq_len)
        gamma = At

        fSCT = torch.mul(fSC, torch.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        fSCT = torch.mean(fSCT, dim=2)  # [bs x 512]
        # print("fSCT 1: ", fSCT.shape)

        if audio.dim() >= 2:
            if audio.size(0) != batch and audio.size(1) == batch:
                audio = audio.transpose(0, 1).contiguous()
        # 그래도 다르면 명확한 에러로 알림
        if audio.size(0) != batch:
            raise RuntimeError(f"Audio batch ({audio.size(0)}) != Visual batch ({batch}). "
                               f"Check process_data_item / dataset collation.")

        # Audio branch
        # print("\n--- Inside forward pass ---")
        # print(f"Before Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")


        audio = (audio - self.audio_mean) / self.audio_std
        # print("audio shape: ", audio.shape)
        

        # print(f"After Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")
        bs = audio.size(0)
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(self.audio_n_segments, dim=0)
        # print("audio shape(2): ", audio)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        audio = torch.unsqueeze(audio, dim=1)
        # print("audio shape(3): ", audio.shape)
        audio = self.a_resnet(audio)
        audio = torch.flatten(audio, start_dim=1).contiguous()
        audio = self.a_fc(audio)
        audio = audio.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        audio = audio.permute(1, 2, 0).contiguous()

        # print("audio shape(4): ", audio.shape)

        Ha = self.aa_net['conv'](audio)
        # print("Ha shape: ", Ha.shape)
        Ha = torch.squeeze(Ha, dim=1)
        # print("Ha shape (2): ", Ha.shape)
        Ha = self.aa_net['fc'](Ha)
        Aa = self.aa_net['relu'](Ha)

        fA = torch.mul(audio, torch.unsqueeze(Aa, dim=1).repeat(1, self.audio_embed_size, 1))
        fA = torch.mean(fA, dim=2)  # [bs x 256]

        # Fusion
        # print("fSCT: ", fSCT.shape)
        # print("fA: ", fA.shape)
        # fSCTA = torch.cat([fSCT, fA], dim=1)
        # output = self.av_fc(fSCTA)

        # Fusion
        # fSCTA = torch.cat([fSCT, fA], dim=1)
        # fA의 배치 크기(32) 중 첫 번째 샘플만 사용합니다.
        # Audio 입력이 모두 동일한 복제본이므로, 어떤 샘플을 사용해도 결과는 같습니다.
        fA_single = fA[0:1, :] # 모양: [32, 256] -> [1, 256]
        # 이제 fSCT와 fA_single 모두 배치 크기가 1이므로 cat이 가능합니다.

        # print("fSCT: ", fSCT.shape)
        # print("fA: ", fA.shape)
        fSCTA = torch.cat([fSCT, fA], dim=1)
        output = self.av_fc(fSCTA)

        return output, alpha, beta, gamma
