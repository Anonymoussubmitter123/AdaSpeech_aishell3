from matplotlib.pyplot import sca
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from collections import OrderedDict


class LayerNorm(torch.nn.Module):
    def __init__(self, nout: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(nout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.transpose(1, -1))
        x = x.transpose(1, -1)
        return x


class UtteranceEncoder(nn.Module):
    """ Acoustic modeling """

    def __init__(self, model_config):
        super(UtteranceEncoder, self).__init__()
        self.idim = model_config["UtteranceEncoder"]["idim"]
        self.n_layers = model_config["UtteranceEncoder"]["n_layers"]
        self.n_chans = model_config["UtteranceEncoder"]["n_chans"]
        self.kernel_size = model_config["UtteranceEncoder"]["kernel_size"]
        self.pool_kernel = model_config["UtteranceEncoder"]["pool_kernel"]
        self.dropout_rate = model_config["UtteranceEncoder"]["dropout_rate"]
        self.stride = model_config["UtteranceEncoder"]["stride"]
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                     nn.Conv1d(
                         self.idim,
                         self.n_chans,
                         self.kernel_size,
                         stride=self.stride,
                         padding=(self.kernel_size - 1) // 2,
                     )
                     ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                     nn.Conv1d(
                         self.n_chans,
                         self.n_chans,
                         self.kernel_size,
                         stride=self.stride,
                         padding=(self.kernel_size - 1) // 2,
                     )
                     ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )

    def forward(self, xs):
        xs = self.conv(xs)
        xs = F.avg_pool1d(xs, xs.size(-1))

        """该代码行使用 PyTorch 中的 F.avg_pool1d 函数对特征进行平均池化，将时域上的维度（即序列长度）压缩为 1。
        因此，最终得到的特征形状为 (B, C, 1)，其中 B 表示 batch size，C 表示特征通道数，1 表示时域上被压缩成了一个点。

        这种池化操作可以将不同长度的输入序列转换为固定长度的特征向量，从而方便后续的分类、回归等任务的处理。
        然而，值得注意的是，平均池化会将输入序列中的信息平均化，可能会丢失一些重要的时序信息，因此在某些情况下，使用其它池化方式，
        如最大池化，可能会更合适。"""

        return xs

    """
    在给定的代码中，UtteranceEncoder 是用于实现句子级别的语音信号编码器，而 PhonemeLevelEncoder 则是用于实现音素级别的语音信号编码器。
    区别在于 UtteranceEncoder 的输入是完整的语音信号，而 PhonemeLevelEncoder 的输入是分段的语音信号（对每个语音片段进行编码）。

    在给定的代码中，输入是通过 forward 函数的 xs 参数传递的。xs 是一个 torch.Tensor 类型的对象，代表语音信号的特征表示。
    具体来说，在 UtteranceEncoder 中，xs 的维度为 (B, C, Tmax)，其中 B 是 batch size，C 是特征维度，Tmax 是时间步数；
    在 PhonemeLevelEncoder 中，xs 的维度为 (B, C, Lmax)，其中 B 是 batch size，C 是特征维度，Lmax 是语音片段的最大长度。
    """


class PhonemeLevelEncoder(nn.Module):
    """ Phoneme level encoder """

    def __init__(self, model_config):
        super(PhonemeLevelEncoder, self).__init__()
        self.idim = model_config["PhonemeLevelEncoder"]["idim"]
        self.n_layers = model_config["PhonemeLevelEncoder"]["n_layers"]
        self.n_chans = model_config["PhonemeLevelEncoder"]["n_chans"]
        self.kernel_size = model_config["PhonemeLevelEncoder"]["kernel_size"]
        self.dropout_rate = model_config["PhonemeLevelEncoder"]["dropout_rate"]
        self.stride = model_config["PhonemeLevelEncoder"]["stride"]
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                     nn.Conv1d(
                         self.idim,
                         self.n_chans,
                         self.kernel_size,
                         stride=self.stride,
                         padding=(self.kernel_size - 1) // 2,
                     )
                     ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                     nn.Conv1d(
                         self.n_chans,
                         self.n_chans,
                         self.kernel_size,
                         stride=self.stride,
                         padding=(self.kernel_size - 1) // 2,
                     )
                     ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        self.linear = nn.Linear(self.n_chans, model_config["PhoneEmbedding"]["phn_latent_dim"])

    def forward(self, xs):
        xs = self.conv(xs)
        xs = self.linear(xs.transpose(1, 2))
        return xs


class PhonemeLevelPredictor(nn.Module):
    """ PhonemeLevelPredictor """

    def __init__(self, model_config):
        super(PhonemeLevelPredictor, self).__init__()
        self.idim = model_config["PhonemeLevelPredictor"]["idim"]
        self.n_layers = model_config["PhonemeLevelPredictor"]["n_layers"]
        self.n_chans = model_config["PhonemeLevelPredictor"]["n_chans"]
        self.kernel_size = model_config["PhonemeLevelPredictor"]["kernel_size"]
        self.dropout_rate = model_config["PhonemeLevelPredictor"]["dropout_rate"]
        self.stride = model_config["PhonemeLevelPredictor"]["stride"]
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                     nn.Conv1d(
                         self.idim,
                         self.n_chans,
                         self.kernel_size,
                         stride=self.stride,
                         padding=(self.kernel_size - 1) // 2,
                     )
                     ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                     nn.Conv1d(
                         self.n_chans,
                         self.n_chans,
                         self.kernel_size,
                         stride=self.stride,
                         padding=(self.kernel_size - 1) // 2,
                     )
                     ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        self.linear = nn.Linear(self.n_chans, model_config["PhoneEmbedding"]["phn_latent_dim"])

    def forward(self, xs):
        xs = self.conv(xs)
        xs = self.linear(xs.transpose(1, 2))

        return xs


class Condional_LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 epsilon=1e-5
                 ):
        super(Condional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = 256
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)  # 求均值
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # 求方差
        std = (var + self.epsilon).sqrt()  # 求标准差，加上epsilon，防止标准差为零
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)

        return y

"""
这段代码实现了一种条件归一化（Conditional Normalization）的操作，这里的条件是指来自说话人的特征。在语音识别等任务中，
不同说话人的语音信号可能有不同的声学特征，因此模型需要在学习中考虑这种条件信息，以提高模型的泛化性能。

这段代码实现了一个条件层归一化（Conditional Layer Normalization）的操作。在神经网络中，层归一化是一种常用的技术，
它可以在深层神经网络中加速训练并提高模型的性能。而条件层归一化是在此基础上进行了改进，它不仅考虑了批次内的统计信息，
还引入了额外的条件信息，例如说说话人的身份，来更好地对模型进行正则化。

具体来说，包括 normal_shape 表示归一化的特征数量，epsilon 表示一个极小值，W_scale 表示归一化的比例因子，W_bias 表示归一化的偏置项。
reset_parameters 方法则初始化了这些线性层的权重和偏置项。

在 forward 方法中，先计算出输入张量 x 在最后一个维度上的平均值 mean 和方差 var，然后通过 std 计算标准差，
并对 x 进行标准化操作得到 y。接着，使用输入的 speaker_embedding 来计算 scale 和 bias，

（在实现中，首先使用一个全连接层(nn.Linear)将输入的speaker embedding转换为一个hidden_size维度的向量，
然后对这个向量进行一些变换来得到scale和bias。具体而言，scale和bias的计算方式如下：

scale：使用一个全连接层(nn.Linear)将speaker embedding转换为一个hidden_size维度的向量，然后对这个向量进行sigmoid激活函数操作，
最后再乘以一个缩放系数gamma（一般初始化为1），得到最终的scale。
bias：使用一个全连接层(nn.Linear)将speaker embedding转换为一个hidden_size维度的向量，直接加上一个偏置项beta（一般初始化为0），
得到最终的bias。

通过这样的计算方式，可以使得在不同说话人的语音数据中，不同维度的特征具有不同的scale和bias值，从而在训练过程中更好地区分不同说话人的特征。）

并将它们分别作用于 y 上，最后得到条件层归一化的结果并返回。

"""