import numpy as np
from aerosonar.config import load_and_merge_configs
from typing import Optional

import torch
import torchaudio


windows = {
    "hann": torch.hann_window,
    "hamming": torch.hamming_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
}


class SpectrogramTransform:
    def __init__(self, config):
        data_config = config["data"]
        spec_config = config["spectrogram"]
        self.duration: int =            data_config.get("duration", None)
        self.sample_rate =              data_config["sample_rate"]
        self.n_fft: int =               spec_config["n_fft"]
        self.win_length: int =          spec_config["win_length"]
        self.hop_length: int =          spec_config["hop_length"]
        self.f_min: float =             spec_config.get("f_min", 0.0)
        self.f_max: Optional[float] =   spec_config.get("f_max", None)
        self.pad: Optional[int] =       spec_config.get("pad", 0)
        self.n_mels: int =              spec_config["n_mels"]
        self.window_fn =                windows[spec_config.get("window", "hann")]
        self.power : float =            spec_config.get("power", 2.0)
        self.eps: float =               spec_config.get("eps", 1e-10)
        self.target_dbfs: float =       spec_config.get("normalize_target_dbfs", -20.0)
        self.max_gain_db: float =       spec_config.get("normalize_max_gain_db", 30.0)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate =   self.sample_rate,
            n_fft =         self.n_fft,
            win_length =    self.win_length,
            hop_length =    self.hop_length,
            f_min =         self.f_min,
            f_max =         self.f_max,
            pad =           self.pad,
            n_mels =        self.n_mels,
            window_fn =     self.window_fn, 
            power =         self.power,
            center =        True,
            pad_mode =      "reflect",
            norm =          None,
            mel_scale =     "htk"
        )

    def _normalize_loudness(self, waveform: torch.Tensor) -> torch.Tensor:
        # Make the spectrogram's absolute level reflect spectral shape rather than
        # mic gain / source distance: scale each chunk to a fixed RMS (dBFS) before
        # the mel transform. Gain is capped so near-silent chunks aren't blown up
        # into amplified noise floor.
        rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        target_rms = 10 ** (self.target_dbfs / 20.0)
        max_gain = 10 ** (self.max_gain_db / 20.0)
        gain = (target_rms / rms).clamp_max(max_gain)
        return waveform * gain

    def __call__(self, waveform):
        waveform = self._normalize_loudness(waveform)
        mel_spec = self.mel_spectrogram(waveform)
        db_mel_spec = self.amplitude_to_db(mel_spec)
        return db_mel_spec
