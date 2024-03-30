import argparse
import torch
import yaml
import os
import sys
import json
import librosa
import numpy as np
from scipy.io import wavfile
from utils.model import get_model, vocoder_infer
from utils.tools import to_device, synth_samples, AttrDict
import audio as Audio

sys.path.append("vocoder")
from vocoder.models.hifigan import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocoder_checkpoint_path = "data/generator_universal.pth.tar"
vocoder_config = "data/config_vn_hifigan.json"


def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()

    return vocoder


def load_mel(mel_path):
    mel_name = os.path.basename(mel_path).split(".")[0]
    mel_spectrogram = torch.load(mel_path)
    mel_spectrogram = torch.from_numpy(mel_spectrogram).to('cuda:0')
    mel_spectrogram = torch.unsqueeze(mel_spectrogram, 0)
    mel_len = mel_spectrogram.shape[2]
    mel_len = torch.tensor([mel_len], dtype=torch.int)
    mel_len = torch.unsqueeze(mel_len, 0)

    return mel_name, mel_len, mel_spectrogram


def synthesize(mel_name, mel_len, mel_spectrogram, vocoder, output_dir):
    lengths = mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(mel_spectrogram, vocoder, model_config, preprocess_config, lengths=lengths)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav in wav_predictions:
        wavfile.write(os.path.join(output_dir, "{}.wav".format(mel_name)), sampling_rate, wav)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/result/cgan/")
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open("config/pretrain/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open("config/pretrain/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("config/pretrain/train.yaml", "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    output_dir = args.output_dir
    STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
    # Load vocoder
    vocoder = get_vocoder(vocoder_config, vocoder_checkpoint_path)

    # Preprocess texts
    mel_name, mel_len, mel_spectrogram = load_mel("raw_data/mel/fake_short_augment_selection203.pt")
    synthesize(mel_name, mel_len, mel_spectrogram, vocoder, output_dir)
    mel_name, mel_len, mel_spectrogram = load_mel("raw_data/mel/fake_short_augment_selection204.pt")
    synthesize(mel_name, mel_len, mel_spectrogram, vocoder, output_dir)
    mel_name, mel_len, mel_spectrogram = load_mel("raw_data/mel/fake_short_augment_selection205.pt")
    synthesize(mel_name, mel_len, mel_spectrogram, vocoder, output_dir)
