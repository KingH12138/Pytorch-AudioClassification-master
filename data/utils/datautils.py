import random
import warnings
import wave

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torchaudio import transforms


warnings.filterwarnings("ignore")


def audio_open(audio_path):
    """
    audio_path -> [tensor:channel*frames,int:sample_rate]
    """
    sig, sr = torchaudio.load(audio_path, channels_first=True)
    return [sig, sr]


def get_wave_plot(wave_path, plot_save_path=None, plot_save=False):
    f = wave.open(wave_path, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    str_bytes_data = f.readframes(nframes=nframes)
    wavedata = np.frombuffer(str_bytes_data, dtype=np.int16)
    wavedata = wavedata * 1.0 / (max(abs(wavedata)))
    time = np.arange(0, nframes) * (1.0 / framerate)
    plt.plot(time, wavedata)
    if plot_save:
        plt.savefig(plot_save_path, bbox_inches='tight')


def regular_channels(audio, new_channels):
    """
    torchaudio-file([tensor,sample_rate])+target_channel -> new_tensor
    """
    sig, sr = audio
    if sig.shape[0] == new_channels:
        return audio
    if new_channels == 1:
        new_sig = sig[:1, :]  # 直接取得第一个channel的frame进行操作即可
    else:
        # 融合(赋值)第一个通道
        new_sig = torch.cat([sig, sig], dim=0)  # c*f->2c*f
    # 顺带提一句——
    return [new_sig, sr]


def regular_sample(audio, new_sr):
    sig, sr = audio
    if sr == new_sr:
        return audio
    channels = sig.shape[0]
    re_sig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
    if channels > 1:
        re_after = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
        re_sig = torch.cat([re_sig, re_after])
    # 顺带提一句torch.cat类似np.concatenate,默认dim=0
    return [re_sig, new_sr]


def regular_time(audio, max_time):
    sig, sr = audio
    rows, len = sig.shape
    max_len = sr // 1000 * max_time

    if len > max_len:
        sig = sig[:, :max_len]
    elif len < max_len:
        pad_begin_len = random.randint(0, max_len - len)
        pad_end_len = max_len - len - pad_begin_len
        # 这一步就是随机取两个长度分别加在信号开头和信号结束
        pad_begin = torch.zeros((rows, pad_begin_len))
        pad_end = torch.zeros((rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)  # 注意哦我们不是增加通道数，所以要制定维度为1
    return [sig, sr]


def time_shift(audio, shift_limit):
    sig, sr = audio
    sig_len = sig.shape[1]
    shift_amount = int(random.random() * shift_limit * sig_len)  # 移动量
    return (sig.roll(shift_amount), sr)


# get Spectrogram
def get_spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = audio
    top_db = 80
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec


def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    return aug_spec

# ----------------------------------------------------------------------------------------------------------------------
def choice_and_process(file_path):
    audio_file = audio_open(file_path)
    audio_file = regular_sample(audio_file, 44100)
    audio_file = regular_channels(audio_file, 2)
    audio_file = regular_time(audio_file, 4000)
    audio_file = time_shift(audio_file, 0.4)
    audio_file = get_spectro_gram(audio_file, n_mels=64, n_fft=1024, hop_len=None)
    audio_file = spectro_augment(audio_file, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    return audio_file