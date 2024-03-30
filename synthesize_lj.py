import argparse
from string import punctuation

import os
import re
import torch
import yaml
import sys
import json
import librosa
import numpy as np
from dataset import TextDataset
from torch.utils.data import DataLoader
from pypinyin import pinyin, Style

# print(torch.cuda.is_available())

# from G2P.convert_text_ipa import convert_text_to_ipa
from utils.model import get_model
from utils.tools import to_device, synth_samples, AttrDict
# from dataset import Dataset
from text import text_to_sequence
# from datetime import datetime
from g2p_en import G2p

import audio as Audio

sys.path.append("vocoder")
from vocoder.models.hifigan import Generator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocoder_checkpoint_path = "vocoder/generator_LJSpeech.pth.tar"
vocoder_config = "data/config_vn_hifigan.json"


def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()

    return vocoder


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text,preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control= control_values
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model.inference(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


def get_reference_mel(reference_audio_dir, STFT):
    wav, _ = librosa.load(reference_audio_dir)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
    return mel_spectrogram


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single", "multiple"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str
    # )
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
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )

    parser.add_argument(
        "--reference_audio",
        type=str
    )

    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )

    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )

    args = parser.parse_args()

    # # Read Config
    # preprocess_config = yaml.load(open("config/AISHELL3/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    # model_config = yaml.load(open("config/pretrain/model.yaml", "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open("config/pretrain/train.yaml", "r"), Loader=yaml.FullLoader)
    # configs = (preprocess_config, model_config, train_config)

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None
    if args.mode == "multiple":
        assert args.source is not None and args.text is None

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    STFT = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(vocoder_config, vocoder_checkpoint_path)

    control_values = args.pitch_control, args.energy_control, args.duration_control

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )

        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

    if args.mode == "single":
        wav_path = args.reference_audio
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        mel_spectrogram = get_reference_mel(wav_path, STFT)
        mel_spectrogram = np.array([mel_spectrogram])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), mel_spectrogram)]

        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

    if args.mode == "multiple":
        with open(args.source, 'r', encoding='utf-8') as file:  # 读取包含多句话的文本文件
            lines = file.readlines()
        texts = []
        speaker_ids = []  # 初始化用于存储句子和speakerid的列表
        ref_wavs = os.listdir(args.reference_audio)  # 初始化reference wavs的列表

        for line in lines:  # 解析每一行，将句子和speakerid分开
            parts = line.strip().split('_')
            if len(parts) == 2:
                text, speaker_id = parts
                texts.append(text)
                speaker_ids.append(int(speaker_id))
            if len(parts) == 1:
                text = parts[0]
                texts.append(text)  # 现在，sentences 列表包含句子，speaker_ids 列表包含相应的 speakerid
                speaker_ids.append(0)

        for i in range(len(texts)):
            # 从文本列表中获取文本和speaker_id和参考音频
            text = texts[i]
            speaker_id = speaker_ids[i]
            ref_wav = ref_wavs[i]

            ids = raw_texts = [text[:100]]
            speakers = np.array([speaker_id])
            if preprocess_config["preprocessing"]["text"]["language"] == "en":
                text = np.array([preprocess_english(text, preprocess_config)])
            elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
                text = np.array([preprocess_mandarin(text, preprocess_config)])
            text_lens = np.array([len(text[0])])
            mel_spectrogram = get_reference_mel(args.reference_audio + ref_wav, STFT)
            mel_spectrogram = np.array([mel_spectrogram])
            batchs = [(ids, raw_texts, speakers, text, text_lens, max(text_lens), mel_spectrogram)]

            synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)