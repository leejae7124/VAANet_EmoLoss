
#saliency 적용
import torch
from opts_dac import parse_opts
import numpy as np
import random

from core.model import generate_model_mean_dac
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset2 import get_training_set, get_validation_set, get_test_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from train import train_epoch
from validation_dac2 import val_epoch

from torch.utils.data import DataLoader
from torch.cuda import device_count

from tensorboardX import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)

    set_seed(42)

    # train
    # spatial_transform = get_spatial_transform(opt, 'train') #여기에서 Preprocessing 객체 생성
    # saliency_transform = get_saliency_transform(opt, 'train', spatial_transform)
    # temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    # target_transform = ClassLabel()
    # training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    # train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    saliency_transform = get_saliency_transform(opt, 'test', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    opt.saliency_level = 'feature_map'

    
    model, _ = generate_model_mean_dac(opt) #중복 어텐션 제거 (1) 모델 생성

    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:0') # GPU 0에 로드
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else: # state_dict만 직접 저장된 경우
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {opt.checkpoint_path}")
    else:
        print("Warning: No checkpoint path provided. Model will run inference on randomly initialized weights or default pretrained weights.")

    
    # --- [디버깅 1] generate_model 호출 직전 opt 값 확인 ---
    print("\n--- Values before calling generate_model ---")

    criterion = get_loss(opt)
    criterion = criterion.cuda()
    # optimizer = get_optim(opt, parameters)

    writer = SummaryWriter(logdir=opt.log_path)

    

    results = val_epoch(0, val_loader, model, criterion, opt, writer)
    print(results)

    writer.close()


if __name__ == "__main__":
    main()

"""
python main.py --expr_name demo
"""