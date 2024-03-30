import torch
import numpy as np
from scipy.io.wavfile import write

from audio.audio_processing import griffin_lim


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    """
    get_mel_from_wav的函数，接受两个参数：audio和_stft，其中_stft是一个声学变换器对象。
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)：使用PyTorch的torch.clip函数将audio张量中的值夹在[-1, 1]范围内，并将其转换为一个浮点型的张量。
    audio = torch.autograd.Variable(audio, requires_grad=False)：将audio张量封装在一个不需要梯度计算的变量中。
    melspec, energy = _stft.mel_spectrogram(audio)：使用声学变换器_stft计算输入音频audio的梅尔频谱和能量谱，并将结果分别赋值给变量melspec和energy。
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)：将梅尔频谱张量的第一个维度（batch维）去除，并将其转换为一个NumPy数组，最后将数据类型转换为np.float32。
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)：将能量谱张量的第一个维度去除，并将其转换为一个NumPy数组，最后将数据类型转换为np.float32
    """

    return melspec, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
    mel = torch.stack([mel])  # 这段代码将 mel 变量的维度从 (80, T) 调整为 (1, 80, T)，其中 80 是频率轴上的维度，T 是时间轴上的维度。
    mel_decompress = _stft.spectral_de_normalize(mel)  # 在将梅尔频谱转换为语音信号时，需要先将梅尔频谱进行非规范化处理，然后通过STFT算法进行逆变换，得到语音信号。
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    # mel_decompress是一个Tensor，其维度是(batch_size, num_mel_channels, T)，
    # 其中batch_size表示批大小，num_mel_channels表示梅尔频率的数量，T表示时域上的帧数。transpose(1, 2)是将维度1和维度2进行交换，
    # 因此mel_decompress的维度变成了(batch_size, T, num_mel_channels)。data.cpu()表示将Tensor从GPU转移到CPU。
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    """
    spec_from_mel_scaling = 1000：将生成的梅尔频谱系数转化为线性幅度时所需要的比例因子，即倍频系数。
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)：将梅尔频谱系数与傅里叶变换的基向量相乘得到线性幅度谱，这里用的是PyTorch的矩阵乘法函数mm。
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)：调整线性幅度谱的维度顺序，使得它的形状变为[batch_size=1, channels=1, time_steps, freq_bins]，这里用的是PyTorch的张量转置函数transpose和扩展维度函数unsqueeze。
    spec_from_mel = spec_from_mel * spec_from_mel_scaling：将线性幅度谱乘上之前定义的比例因子，以得到最终的音频幅度值。
    """

    audio = griffin_lim(
        torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
    )

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, _stft.sampling_rate, audio)
    """
    audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters)：使用Griffin-Lim算法，
    将输入的mel频谱还原为音频信号。其中，spec_from_mel[:, :, :-1]表示将最后一列去掉，这是因为Griffin-Lim算法需要输入的频谱是实数值，而mel频谱是复数值，因此需要丢掉最后一列。
    
    audio = audio.squeeze()：将维度为1的维度去掉，即从(1, length)变为(length,)。
    audio = audio.cpu().numpy()：将张量转换为numpy数组，并从GPU内存中移动到CPU内存中。
    audio_path = out_filename：将输出路径保存到audio_path变量中。
    write(audio_path, _stft.sampling_rate, audio)：使用soundfile库将音频信号写入到指定路径audio_path中。_stft.sampling_rate是音频的采样率。
    """
