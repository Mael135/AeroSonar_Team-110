"""Waveform-to-spectrogram feature extraction.

Defines the single transform used by preprocessing, training and live inference, so
that all three see identically scaled inputs.
"""
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
    """Converts a waveform into a loudness-normalised decibel mel spectrogram.

    The transform is configured entirely from the project config, reading the ``data``
    section for the sample rate and clip duration and the ``spectrogram`` section for
    the mel parameters.

    Call the instance to run the full chain: loudness normalisation, mel spectrogram,
    then amplitude-to-decibel conversion.

    Attributes:
        mel_spectrogram: The underlying ``torchaudio`` mel transform. Exposed so
            callers can move it to a specific device.
        amplitude_to_db: The decibel conversion module, likewise movable.
        target_dbfs: RMS level each clip is normalised to.
        max_gain_db: Ceiling on the normalisation gain.
    """

    def __init__(self, config):
        """Build the transform from a configuration dictionary.

        Args:
            config: Full project configuration, containing ``data`` and
                ``spectrogram`` sections.
        """
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
        """Scale a waveform to a fixed RMS level.

        Without this step the spectrogram's absolute level encodes microphone gain,
        operating-system input volume and source distance rather than spectral
        content. Gain is capped at ``max_gain_db`` so that near-silent clips are not
        amplified into noise.

        Args:
            waveform: Shape ``(channels, samples)``. RMS is computed per channel.

        Returns:
            torch.Tensor: The rescaled waveform, same shape as the input.
        """
        rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
        target_rms = 10 ** (self.target_dbfs / 20.0)
        max_gain = 10 ** (self.max_gain_db / 20.0)
        gain = (target_rms / rms).clamp_max(max_gain)
        return waveform * gain

    def __call__(self, waveform):
        """Run the full feature-extraction chain.

        Args:
            waveform: Shape ``(channels, samples)``.

        Returns:
            torch.Tensor: Decibel mel spectrogram of shape
            ``(channels, n_mels, frames)``, where ``frames`` is
            ``samples // hop_length + 1`` for the centred STFT.
        """
        waveform = self._normalize_loudness(waveform)
        mel_spec = self.mel_spectrogram(waveform)
        db_mel_spec = self.amplitude_to_db(mel_spec)
        return db_mel_spec
