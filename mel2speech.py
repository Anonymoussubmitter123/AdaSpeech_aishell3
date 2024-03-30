import argparse
from PIL import Image
import numpy as np
import torch
import librosa
import os
import yaml
import json
from utils.model import vocoder_infer
from utils.tools import to_device, log, synth_one_sample, AttrDict
import sys
sys.path.append("vocoder")
from vocoder.models.hifigan import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义路径和文件名
image_file_path = "raw_data/real_mel_data/400dpi-1/SSB03950002.png"
vocoder_checkpoint_path = "vocoder/generator_universal.pth.tar"
vocoder_config_path = "vocoder/config/config_v1.json"


def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()

    return vocoder


def img2wav(image, _stft):
    # 将 MEL 频谱转换回音频
    image = torch.autograd.Variable(image, requires_grad=False)
    audio = _stft.spectral_de_normalize(image)
    audio = torch.pow(10.0, audio)
    audio = _stft.spectral_post_inverse(audio)
    audio = audio.transpose(1, 2).detach().cpu().numpy()[0]

    return audio


def mel2wav(mel, vocoder, model_config, preprocess_config):
    wav = vocoder_infer(
                mel.squeeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]

    return wav


def main(args, configs):
    preprocess_config, model_config = configs

    image = Image.open(image_file_path).convert('L')  # 读取png图片L并转化成灰度图像
    image = np.array(image)
    image = image / 255.0  # 图像归一化，将图像像素值缩放到0-1之间
    image = torch.from_numpy(image).unsqueeze(0)
    print(image.shape)
    audio = img2wav(image)

    mel_spec = librosa.feature.melspectrogram(
        y=None,
        sr=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        S=image,
        n_fft=preprocess_config["preprocessing"]["stft"]["filter_length"],
        hop_length=preprocess_config["preprocessing"]["stft"]["hop_length"],
        n_mels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        fmin=preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        fmax=preprocess_config["preprocessing"]["mel"]["mel_fmax"])

    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    print(mel_spec.shape)
    mel_spec = torch.squeeze(mel_spec, 0).numpy().astype(np.float32)
    print(mel_spec.shape)

    vocoder = get_vocoder(vocoder_config_path, vocoder_checkpoint_path)
    wav = mel2wav(mel_spec, vocoder, model_config, preprocess_config)
    basename = image.split('/')[-1].split('.')[0]
    output_path = os.path.join("raw_data/mel2wav_output", '{}.wav'.format(basename))
    torch.save(wav, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config)

    main(args, configs)