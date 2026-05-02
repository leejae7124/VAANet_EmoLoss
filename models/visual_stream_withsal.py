import torch
import torch.nn as nn
import torchvision
from models.resnet import pretrained_resnet101


class VisualStream(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_resnet101_path):
        super(VisualStream, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        self.pretrained_resnet101_path = pretrained_resnet101_path

        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_attention_subnets()
        self._init_params()

    def _init_norm_val(self):
        self.NORM_VALUE = 255.0 #이미지 정규화할 때 흔히 사용되는 "최대 픽셀값"
        self.MEAN = 100.0 / self.NORM_VALUE #mean은 정규화 과정에서 뺄 평균값을 의미. 100을 픽셀 평균값으로 사용하겠다.

    def _init_encoder(self):
        resnet, _ = pretrained_resnet101(snippet_duration=self.snippet_duration,
                                         sample_size=self.sample_size,
                                         n_classes=self.n_classes,
                                         ft_begin_index=self.ft_begin_index,
                                         pretrained_resnet101_path=self.pretrained_resnet101_path)

        children = list(resnet.children())
        self.resnet = nn.Sequential(*children[:-2])  # delete the last fc and the avgpool layer
        for param in self.resnet.parameters():
            param.requires_grad = False

    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):
        self.conv0 = nn.Sequential(
            *[nn.Conv1d(self.hp['nc'], self.hp['k'], 1, bias=True),
              nn.BatchNorm1d(self.hp['k']),
              nn.ReLU()])

        self.sa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.hp['k'], 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.hp['m'], self.hp['m'], bias=False),
            'softmax': nn.Softmax(dim=1)
        })

        self.ta_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.hp['k'], 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.seq_len, self.seq_len, bias=True),
            'relu': nn.ReLU()
        })

        self.cwa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.hp['m'], 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.hp['k'], self.hp['k'], bias=False),
            'softmax': nn.Softmax(dim=1)
        })

        self.fc = nn.Linear(self.hp['k'], self.n_classes) #시각 특징을 받아서 8개의 감정 클래스로

    def _init_params(self):
        for subnet in [self.conv0, self.sa_net, self.ta_net, self.cwa_net, self.fc]:
            if subnet is None:
                continue
            for m in subnet.modules():
                self._init_module(m)
        self.ta_net['fc'].bias.data.fill_(1.0)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, input: torch.Tensor):
        input = input.transpose(0, 1).contiguous()  # transpose 이후, input.shape=[seq_len, batch, 3, 16, 112, 112]
        #0, 1번째 값을 서로 바꾸고(transpose), 
        #contiguouse(): 텐서가 물리적으로 메모리에 연속적으로 저장되도록 강제.
        # -> transpose를 사용하고 나면 input 데이터는 읽는 방식을 바꾸어 내가 원하는 shape로 읽을 수 있지만,
        # 실제 메모리는 이전 순서로 저장된 데이터에 접근할 수 있다. 따라서, 새로운 메모리 공간을 할당하고, 원본 데이터를 이 새로운 메모리 공간에 연속적으로 복사한다.
        # 결론: transpose, contiguous는 세트이다.
        input.div_(self.NORM_VALUE).sub_(self.MEAN)
        #_(언더스코어)가 붙은 함수는 해당 텐서 자체를 수정한다. (input 텐서 값을 변경함)
        #div_: input 텐서의 모든 픽셀 값을 self.NORM_VALUE(255.0)으로 나눈다. - 0~1 사이로 스케일링
        #왜 딥러닝에서 스케일링이 중요한가?
        #기울기 소실/폭주 방지. 입력값이 너무 크거나 작으면 활성 함수의 포화(saturation), wmr rldnfrlrk 0dp rkRkdnjwlseksms thfl!!
        #기울기 소실: 활성화 함수의 입력이 포화 영역에 도달하면, 해당 지점 기울기가 거의 0이 된다. -> 가중치 업데이트가 거의 안 일어남
        #기울기 폭주: 입력 값이 너무 크거나 가중치가 너무 크면, 역전파 시 기울기가 급격히 커지는 기울기 폭주 문제 발생. -> 업데이트가 너무 크게 일어나 최적점을 지나쳐 발산함.

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        # view: 텐서 모양(shape)을 변경하는 파이토치 함수. 여기서는 seq_len과 batch를 곱하여서 새로운 첫 번째 차원으로 만든다.
        # -> 이 재구조화는 일반적으로 비디오나 3D 데이터를 이미지 처리 모델에 입력하기 위해 수행된다.
        
        with torch.no_grad(): # 파이토치의 자동 미분(autograd) 시스템을 비활성화. -> 가중치 업데이트 필요 없을 때, 여기서 resnet 이미 학습됨!!
            output = self.resnet(input) #resnet을 input을 입력으로 받는다.
            output = torch.squeeze(output, dim=2) #dim=2는 3번째 차원. snippet_duration이 1이 되어, 그 차원을 제거한다.
            output = torch.flatten(output, start_dim=2)
            # 하나의 스니펫을 받아, 이 스니펫 전체를 아우르는 단일 특징 벡터를 출력한다. 가장 중요한 시공간적 특징을 뽑아낸다.
        F = self.conv0(output)  # [B x 512 x 16]

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

        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = torch.mul(fSC, torch.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        fSCT = torch.mean(fSCT, dim=2)

        output = self.fc(fSCT)
        return output, alpha, beta, gamma
