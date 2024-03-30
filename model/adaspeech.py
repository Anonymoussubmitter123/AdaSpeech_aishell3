from distutils.command.config import config
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .adaspeech_modules import UtteranceEncoder, PhonemeLevelEncoder, PhonemeLevelPredictor, Condional_LayerNorm
from utils.tools import get_mask_from_lengths


class AdaSpeech(nn.Module):
    """ AdaSpeech """

    def __init__(self, preprocess_config, model_config):
        super(AdaSpeech, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.UtteranceEncoder = UtteranceEncoder(model_config)
        self.PhonemeLevelEncoder = PhonemeLevelEncoder(model_config)
        self.PhonemeLevelPredictor = PhonemeLevelPredictor(model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.speaker_emb = nn.Embedding(
            model_config["language_speaker"]["num_speaker"],
            model_config["transformer"]["encoder_hidden"]
        )
        self.phone_level_embed = nn.Linear(
            model_config["PhoneEmbedding"]["phn_latent_dim"],
            model_config["PhoneEmbedding"]["adim"]
        )
        # self.lang_emb = nn.Embedding(
        #     model_config["language_speaker"]["num_language"],
        #     model_config["transformer"]["encoder_hidden"]
        # )
        self.layer_norm = Condional_LayerNorm(preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        self.postnet = PostNet()

    def forward(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            avg_targets=None,
            languages=None,
            phoneme_level_predictor=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        speaker_embedding = self.speaker_emb(speakers)
        # language_embedding = self.lang_emb(languages)
        output = self.encoder(texts, speaker_embedding, src_masks)
        xs = self.UtteranceEncoder(torch.transpose(mels, 1, 2))
        xs = torch.transpose(xs, 1, 2)
        output = output + xs.expand(-1, max_src_len, -1)

        if phoneme_level_predictor:
            phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
            with torch.no_grad():
                phn_encode = self.PhonemeLevelEncoder(avg_targets.transpose(1, 2))
            output = output + self.phone_level_embed(phn_encode.detach())
        else:
            phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
            phn_encode = self.PhonemeLevelEncoder(avg_targets.transpose(1, 2))
            output = output + self.phone_level_embed(phn_encode)
        """
        判断是训练还是推理
        
        它可以暂时禁用 PyTorch 中的梯度计算，也就是说在这个上下文中计算的所有张量不会记录梯度，不会影响模型的训练。
        通常，在模型推理（即使用模型进行预测而不是更新模型参数）时使用 with torch.no_grad(): 可以提高推理的效率，
        并且可以减少内存的使用。因为在模型推理过程中，我们不需要计算每个参数的梯度，只需要前向传递计算输出即可。
        """

        output = output + speaker_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        # output = output + language_embedding.unsqueeze(1).expand(
        #     -1, max_src_len, -1
        # )
        """
        具体来说，unsqueeze(1) 将张量从 (batch_size, embedding_size) 变成 (batch_size, 1, embedding_size)，
        然后 expand(-1, max_src_len, -1) 将第二个维度从 1 扩展到 max_src_len，变成 (batch_size, max_src_len, embedding_size)。
        这样，language_embedding 张量就被扩展到和output 张量相同的形状，可以进行相加。
        """
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, speaker_embedding, mel_masks)
        output = self.mel_linear(output)
        output = self.layer_norm(output, speaker_embedding)
        """
        这段代码是调用 nn.LayerNorm 对输入 output 进行 Layer Normalization，以减少模型训练过程中的内部协变量偏移问题，
        提高模型的泛化能力。nn.LayerNorm 是一个标准的 PyTorch 内置函数，它的第一个输入是需要进行归一化的张量，第二个输入是
        用于归一化的均值和标准差。这里的 speaker_embedding 是对输入 output 进行 speaker-aware normalization 所需要的信息，
        以更好地将输出与说话人特定的信息区分开来。
        
        在 BatchNormalization 中，我们有一个输入 x（大小为 m×n），需要对它进行归一化。具体来说，对于每个特征维度（即列），
        我们要将其数据分布变成均值为 0、标准差为 1 的标准正态分布。因此，我们需要求出 x 在每个特征维度上的均值和标准差，
        然后使用这些值来对 x 进行归一化。假设 x 在第 i 个特征维度上的均值为 μi，标准差为 σi。那么对于 x 中的每个元素 x_ij，
        它的归一化结果为：(x_ij - μi) / σi.但是，直接对 x 进行归一化存在一个问题：它的每个元素都受到其他元素的影响。
        也就是说，如果 x 中的某个元素在更新之后，那么它所在的特征维度上的均值和标准差也会随之发生变化，从而影响到其他元素的归一化结果。
        这种现象被称为“内部协变偏移”。BatchNormalization 的解决方案是引入可学习的参数 γ 和 β。在归一化之后，我们将结果乘以 γ 并加上 β，
        从而使得每个特征维度上的数据分布可以根据数据本身进行调整。因此，归一化结果为：
                        y_ij = γi * (x_ij - μi) / σi + βi
        其中，γi 和 βi 都是需要学习的参数。在训练过程中，它们会被不断更新，以适应不同的数据分布。
        """
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            phn_predict,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phn_encode,
        )

    def inference(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            # languages=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            avg_targets=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        speaker_embedding = self.speaker_emb(speakers)
        # language_embedding = self.lang_emb(languages)
        output = self.encoder(texts, speaker_embedding, src_masks)
        xs = self.UtteranceEncoder(torch.transpose(mels, 1, 2))
        xs = torch.transpose(xs, 1, 2)
        output = output + xs.expand(-1, max_src_len, -1)

        phn_predict = self.PhonemeLevelPredictor(output.transpose(1, 2))
        phn_encode = None
        output = output + self.phone_level_embed(phn_predict)

        output = output + speaker_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )

        # output = output + language_embedding.unsqueeze(1).expand(
        #     -1, max_src_len, -1
        # )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, speaker_embedding, mel_masks)
        output = self.mel_linear(output)
        output = self.layer_norm(output, speaker_embedding)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            phn_predict,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            phn_encode,
        )
