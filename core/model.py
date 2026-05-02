import torch.nn as nn
from models.visual_stream import VisualStream
from models.vaanet_saliency_prior_att import VAANet_sal_bias
from models.vaanet import VAANet
from models.vaanet_saliency1 import VAANet2 #기존 중복 어텐션
from models.vaanet_saliency_remove1 import VAANet_remove1 #중복 어텐션 제거 (1) -> 기존 방식과 다르게 가중치 적용함
from models.vaanet_saliency_only import VAANet_saliency_only # 중복 어텐션 제거: 기존 방식대로 적용.

from models.vaanet_intensity import VAANet_intensity

from models.vaanet_test_origin import VAANet_test
from models.vaanet_saliency1_test import VAANet2_test



def generate_model(opt):
    model = VAANet(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        # audio_mean=opt.audio_mean,
        # audio_std=opt.audio_std, #오디오 정규화로 필요
        saliency_level=opt.saliency_level
    )
    # model = VisualStream(
    #     snippet_duration=opt.snippet_duration,
    #     sample_size=opt.sample_size,
    #     n_classes=opt.n_classes,
    #     seq_len=opt.seq_len,
    #     pretrained_resnet101_path=opt.resnet101_pretrained,
    # )
    model = model.cuda()
    return model, model.parameters()

def generate_model_remove1(opt):
    model = VAANet_remove1(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        # audio_mean=opt.audio_mean,
        # audio_std=opt.audio_std,
        saliency_level=opt.saliency_level
    )
    model = model.cuda()
    return model, model.parameters()

def generate_model2(opt):
    model = VAANet2(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        # audio_mean=opt.audio_mean,
        # audio_std=opt.audio_std,
        saliency_level=opt.saliency_level
    )
    model = model.cuda()
    return model, model.parameters()

def generate_model_saliency_only(opt):
    model = VAANet_saliency_only(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        # audio_mean=opt.audio_mean,
        # audio_std=opt.audio_std,
        saliency_level=opt.saliency_level
    )
    model = model.cuda()
    return model, model.parameters()



def generate_model_sal_bias(opt):
    model = VAANet_sal_bias(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    model = model.cuda()
    return model, model.parameters()



def generate_model_intensity(opt):
    model = VAANet_intensity(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained
    )
    model = model.cuda()
    return model, model.parameters()

def generate_model2_test(opt):
    model = VAANet2_test(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    model = model.cuda()
    return model, model.parameters()



def generate_model_test(opt):
    model = VAANet_test(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        saliency_level=opt.saliency_level
    )
    model = model.cuda()
    return model, model.parameters()