import torch
import torch.nn as nn
import torchvision
from models.visual_stream import VisualStream

#gradient 때문에 test 코드를 따로 작성함.
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

    def forward(self, visual: torch.Tensor, audio: torch.Tensor, saliency_map: torch.Tensor):
        print(f"🔍 saliency_map.shape = {saliency_map.shape}")
        print(f"🔍 visual.shape = {visual.shape}")
        print(f"🔍 audio.shape = {audio.shape}")
        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        print(f"[INPUT] visual (Seq,B,C,D,H,W): {visual.shape}")
        print(f"[INPUT] saliency_map (B,Seq,1,D,H,W): {saliency_map.shape}")
        # 추가: input level saliency
        if self.saliency_level == 'input':
            # saliency_map: [B, Seq, 1, D, H, W] → [Seq, B, 1, D, H, W] 📦 [Train] saliency_map shape: torch.Size([32, 12, 1, 16, 112, 112])
            saliency_map = saliency_map.transpose(0, 1).contiguous()  # [Seq, B, 1, D, H, W]
            print("saliency map shape(input): ", saliency_map.shape)
            # visual: [Seq, B, C, D, H, W]
            saliency_map = saliency_map.to(visual.device, dtype=visual.dtype)
            print(f"[INPUT] saliency_map after transpose (Seq,B,1,D,H,W): {saliency_map.shape}")
            saliency_mask = saliency_map.expand_as(visual)  # [Seq, B, C, D, H, W] saliency의 채널 차원을 visual과 동일하게 맞춤
            print(f"[INPUT] saliency_mask_v expand_as visual: {saliency_mask.shape}")
            # Residual 방식과 유사한 soft masking (중요한 영역 강조가 아닌, 덜 중요한 영역 감쇠의 방식)
            visual = 0.5 * visual + (1 - 0.5) * (visual * saliency_mask)
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        # with torch.no_grad():
        # ResNet을 통과시켜 피처맵 생성
        F = self.resnet(visual)
        # --- [추가] Feature Map 레벨 Saliency 적용 로직 ---
        if self.saliency_level == 'feature_map':
            saliency_map = saliency_map.transpose(0, 1).contiguous()
            saliency_map_flat = saliency_map.view(seq_len * batch, 1, snippet_duration, sample_size, sample_size)
            print("saliency map shape(feature_map): ", saliency_map.shape)
            print("saliency map flat shape(feature_map): ", saliency_map_flat.shape)
            # saliency map 크기를 feature map에 맞게 리사이즈
            saliency_map_flat = saliency_map_flat.to(F.device, dtype=F.dtype)
            saliency_resized = nn.functional.adaptive_avg_pool3d(saliency_map_flat, (F.size(2), F.size(3), F.size(4)))
            
            # 리사이즈 후
            print("saliency_resized:", saliency_resized.shape)  # 기대: [384, 1, T', H', W']
            # saliency_map shape: [B, Seq, 1, D, H, W] -> [B*Seq, 1, D, H, W]
            
            # 채널 차원으로 확장하여 마스크 생성
            saliency_mask = saliency_resized.expand_as(F)
            print("saliency_mask == F:", saliency_mask.shape, F.shape)  # 동일해야 OK
            
            # Saliency 적용
            F = 0.5 * F + (1 - 0.5) * (F * saliency_mask)
            
        F = torch.squeeze(F, dim=2)
        F = torch.flatten(F, start_dim=2)
        print("F after squeeze/flatten:", F.shape)  # [384, C, T'*H'*W']
        F = self.conv0(F)  # [B x 512 x 16]

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
        print(f"[TA] fSC: {tuple(fSC.shape)} (expect [B, K, Seq])")
        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        print(f"[TA] Ht before fc: {tuple(Ht.shape)}, fc.in={self.ta_net['fc'].in_features}")
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



        # audio = (audio - self.audio_mean) / self.audio_std
        print("audio shape: ", audio.shape)

        print(f"After Normalization - Min: {torch.min(audio)}, Max: {torch.max(audio)}")
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
