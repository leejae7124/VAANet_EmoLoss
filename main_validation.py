from opts import parse_opts

from core.model import generate_model2
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, get_saliency_transform
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from train import train_epoch
from validation_f1 import val_epoch

from torch.utils.data import DataLoader
from torch.cuda import device_count
import torch
import os
import glob



def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    saliency_transform = get_saliency_transform(opt, 'test', spatial_transform)
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, saliency_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)
    
    # audio_mean, audio_std = -6.4479, 68.5871 #split 1
    # audio_mean, audio_std = -6.613560199737549, 70.09141540527344 #split2
    # audio_mean, audio_std = -6.46166753768920, 68.90550231933594 #split3
    # audio_mean, audio_std = -6.570328712463379, 69.36627197265625 #split4
    audio_mean, audio_std = -6.5693535804748535, 69.82913208007812 #split5
    opt.audio_mean = audio_mean # tensor에서 float 값으로 변환
    opt.audio_std = audio_std
    # opt.saliency_level = 'feature_map'
    opt.saliency_level = 'input'
    print(f"Calculated Audio Stats -> Mean: {opt.audio_mean:.4f}, Std: {opt.audio_std:.4f}")

    print(f"opt.audio_mean: {opt.audio_mean}")
    print(f"opt.audio_std: {opt.audio_std}\n")
    model, parameters = generate_model2(opt)

    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    resume = "./real_results/input/split5_save_80_0.3894.pth"
    state = torch.load(resume, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = state.get('state_dict', state)
    model.load_state_dict(state_dict, strict=True)
    print(f"[Eval] Loaded checkpoint: {resume}")

    

    val_epoch(1, val_loader, model, criterion, opt)



if __name__ == "__main__":
    main()

"""
python main.py --expr_name demo
"""