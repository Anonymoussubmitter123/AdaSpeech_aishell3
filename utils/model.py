import os
import json

import torch
import numpy as np

from model import AdaSpeech, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = AdaSpeech(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # print("Model src_word_emb weight shape:", model.encoder.src_word_emb.weight.shape)
        # for key, param in ckpt.items():
        #     if key.startswith('lang_emb'):
        #         ckpt[key[7:]] = param
        #         ckpt.pop(key)
        #         torch.load_state_dict(ckpt)
        # state_dict = model.state_dict()
        # for name, param in ckpt.items():
        #     if name not in state_dict:
        #         continue
        #     if isinstance(param, torch.nn.Parameter):
        #         # backwards compatibility for serialized parameters
        #         param = param.data
        #     state_dict[name].copy_(param)
        # ckpt["model"]['encoder.src_word_emb.weight'].resize_(110, 256)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def load_pretrain(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = AdaSpeech(preprocess_config, model_config).to(device)
    ckpt_path = args.pretrain_dir
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    scheduled_optim = ScheduledOptim(
        model, train_config, model_config, 0
    )
    for param in model.named_parameters():
        if "layer_norm" not in param[0]:
            param[1].requires_grad = False
        if "encoder" in param[0]:
            param[1].requires_grad = False
        if "variance_adaptor" in param[0]:
            param[1].requires_grad = False
        if "UtteranceEncoder" in param[0]:
            param[1].requires_grad = False
        if "PhonemeLevelEncoder" in param[0]:
            param[1].requires_grad = False
        if "PhonemeLevelPredictor" in param[0]:
            param[1].requires_grad = False
        if "speaker_emb" in param[0]:
            param[1].requires_grad = True
    model.train()
    return model, scheduled_optim


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    """在这个函数中，mels是一个大小为 (B, C, T) 的张量，表示B个音频样本的Mel频谱，其中C是Mel频谱的通道数，通常为80或者128，
    T是Mel频谱的时间步数，即每个Mel频谱的宽度。这个函数的作用是将这些Mel频谱转换成音频信号（.wav）格式。"""
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
