import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from utils import get_params
import lws


hparams = get_params('./params/audio_params.json')

def mu_law_encode(audio, quantization_channels = 256):
    '''
    Arguments:
        audio: type(np.array), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        np.array of shape(sequence_length)
    Return:
        np.array with each element ranging from 0 to 255
        The size of return array is the same as input tensor
    '''
    mu = quantization_channels - 1
    safe_audio_abs = np.abs(np.clip(audio, -1.0, 1.0))
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    encoded = (signal + 1) / 2 * mu + 0.5
    return encoded.astype(np.float32)

def mu_law_decode(output, quantization_channels = 256):
    '''
    Arguments:
        audio: type(np.array), size(sequence_length)
        quantization_channels: as the name describes
    Input:
        np.array of shape(sequence_length)
    Return:
        np.array with each element ranging from 0 to 255
        The size of return array is the same as input tensor
    '''
    mu = quantization_channels - 1
    signal = 2.0 * (output / mu) - 1.0
    magnitude = (1.0 / mu) * ((1.0 + mu) ** np.abs(signal) - 1.0)
    return np.sign(signal) * magnitude

def load_wav(path):
    return librosa.core.load(path, sr=hparams["sample_rate"], mono=True)[0]


def save_wav(wav, path):
    #wavfile.write(path, hparams["sample_rate"], wav.astype(np.int16))
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams["sample_rate"], wav.astype(np.int16))

def spectrogram(y):
    D = _lws_processor().stft(y).T
    S = _amp_to_db(np.abs(D)) - hparams["ref_level_db"]
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + hparams["ref_level_db"])  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** hparams["power"])
    y = processor.istft(D).astype(np.float32)
    return y

def spectrogram1(y):
    return _lws_processor().stft(y).T

def inv_spectrogram1(spectrogram):
    D = spectrogram
    processor = _lws_processor()
    D = processor.run_lws(D.astype(np.float64).T)
    y = processor.istft(D).astype(np.float32)
    return y

def melspectrogram(y):
    D = _lws_processor().stft(y).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams["ref_level_db"]
    if not hparams["allow_clipping_in_normalization"]:
        assert S.max() <= 0 and S.min() - hparams["min_level_db"] >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(hparams["fft_size"], hparams["hop_size"], mode="music", perfectrec=True)


# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if hparams["fmax"] is not None:
        assert hparams["fmax"] <= hparams["sample_rate"] // 2
    return librosa.filters.mel(hparams["sample_rate"], hparams["fft_size"],
                               fmin=hparams["fmin"], fmax=hparams["fmax"],
                               n_mels=hparams["num_mels"])


def _amp_to_db(x):
    min_level = np.exp(hparams["min_level_db"] / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams["min_level_db"]) / -hparams["min_level_db"], 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams["min_level_db"]) + hparams["min_level_db"]