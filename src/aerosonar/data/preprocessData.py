"""Offline preprocessing of raw recordings into spectrogram tensors.

Segments each raw WAV file into fixed-length chunks, converts every chunk to a decibel
mel spectrogram, and writes it as an individual ``.pt`` tensor alongside two metadata
files:

``metadata.csv``
    ``filename``, ``target``, ``file_id`` — the training index. ``file_id`` identifies
    the source recording and is what the recording-level split operates on.
``expanded_metadata.csv``
    ``filename`` plus every field parsed from the source filename, including
    ``location``, which the split and the per-location evaluation join against.

Run from the repository root::

    python -m aerosonar.data.preprocessData
"""
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as transforms
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.config import load_default_config


def mask_spec_sfm(spectrogram: torch.Tensor, threshold: float = 0.999999463558197) -> int:
    """Flag a chunk as tonal using spectral flatness.

    Spectral flatness is the ratio of the geometric to the arithmetic mean of the
    power spectrum. Tonal content such as rotor harmonics lowers it; broadband noise
    keeps it near one. The 10th percentile across frames is used rather than the mean,
    because a distant drone is only tonal in a minority of frames.

    Args:
        spectrogram: Decibel spectrogram of shape ``(freq_bins, frames)``.
        threshold: Flatness below which the chunk counts as tonal.

    Returns:
        int: 1 if the chunk is tonal, 0 otherwise.

    Note:
        Diagnostic helper, not used by the preprocessing pipeline. Prints the measured
        value as a side effect. See :mod:`aerosonar.utils.audit` for the maintained
        implementation.
    """
    # Convert dB to Amplitude (Power 1.0 is Amplitude, 2.0 is Power)
    # Most SFM formulas use Power Spectrograms.
    amp_spec = F.DB_to_amplitude(spectrogram, ref=1.0, power=2.0)

    eps = 1e-10
    # SFM calculation
    log_spec = torch.log(amp_spec + eps)
    g_mean = torch.exp(torch.mean(log_spec, dim=0))
    a_mean = torch.mean(amp_spec, dim=0)

    sfm_per_frame = g_mean / (a_mean + eps)

    val_to_check = torch.quantile(sfm_per_frame, 0.1).item()
    print(val_to_check)
    return 1 if val_to_check < threshold else 0



def mask_spec_prominence(spectrogram: torch.Tensor, db_threshold: float = 30) -> int:
    """Flag a chunk by peak-to-median level difference.

    The gap in decibels between the loudest and the median bin acts as a
    signal-to-noise proxy for a prominent source.

    Args:
        spectrogram: Decibel spectrogram of shape ``(freq_bins, frames)``.
        db_threshold: Minimum difference, in decibels, to flag the chunk.

    Returns:
        int: 1 if a prominent source is present, 0 otherwise.

    Note:
        Diagnostic helper, not used by the preprocessing pipeline. Prints the measured
        value as a side effect.
    """
    # This works well directly on dB Mel-spectrograms
    # Calculate the max intensity bin vs the median intensity bin across the 2s chunk
    max_val = torch.max(spectrogram)
    median_val = torch.median(spectrogram)

    diff = max_val - median_val
    print(diff)
    return 1 if diff.item() > db_threshold else 0



def mask_spec_variance(spectrogram: torch.Tensor, var_threshold: float = 100) -> int:
    """Flag a chunk by the temporal stability of its energy.

    High variance suggests a transient such as speech or an impact rather than the
    steady output of a drone.

    Args:
        spectrogram: Decibel spectrogram of shape ``(freq_bins, frames)``.
        var_threshold: Variance below which the chunk counts as steady.

    Returns:
        int: 1 if the energy is steady, 0 otherwise.

    Note:
        Diagnostic helper, not used by the preprocessing pipeline. Prints the measured
        value as a side effect.
    """
    time_energy = torch.mean(spectrogram, dim=0) # Average over Mel-bins per time frame
    variance = torch.var(time_energy)
    print(variance.item())
    return 1 if variance.item() < var_threshold else 0



def parse_filename_metadata(filename: str):
    """Extract recording metadata encoded in a raw filename.

    Filenames follow the convention::

        DATE__LOCATION__LABEL__NOISE__gGAIN__SRkNN__DURATION__INDEX.wav

    for example ``2026-01-03__Yarkon__drone__high-noise_road__g75__22k05__5m__01.wav``.
    Sample rate ``22k05`` parses to 22050 Hz; duration ``1m30`` parses to 90 seconds.

    Args:
        filename: Filename or path. Only the stem is inspected.

    Returns:
        dict: Keys ``is_drone``, ``date``, ``location``, ``noise_level``,
        ``noise_type``, ``gain``, ``sr``, ``duration`` (seconds) and ``num``.

    Raises:
        ValueError: If the stem does not split into exactly eight
            double-underscore-separated fields.
    """
    date, location, label, noise, gain, sr, duration, num = Path(filename).stem.split('__')
    is_drone = True if label == 'drone' else False
    location = location.replace("_", " ")
    if '_' in noise and len(noise.split('_')) == 2:
        parts = noise.split('_')
        noise_level = parts[0].replace('-', ' ')
        noise_type =  parts[1].replace('-', ', ')
    else:
        noise_type = noise
        noise_level = ''
    gain = int(gain.replace('g', ''))
    sr = int(1000 * float(sr.replace('k', '.')))
    duration = duration.split('m')
    if len(duration) > 1:
        m, s = duration
        duration = (60 * int(m)) + int(s) if s != '' else (60 * int(m))
    else:
        duration = int(duration[0])
    num = int(num)
    return {"is_drone": is_drone,
            "date": date,
            "location": location,
            "noise_level": noise_level,
            "noise_type": noise_type,
            "gain": gain,
            "sr": sr,
            "duration": duration,
            "num": num
            }

def process_data():
    """Convert every raw recording into spectrogram tensors and metadata files.

    Reads WAV files from ``paths.data_raw``, splits each into non-overlapping chunks of
    ``data.duration`` seconds, applies
    :class:`~aerosonar.features.transforms.SpectrogramTransform`, and writes one tensor
    per chunk to ``paths.data_processed`` as ``drone_NNNNNN.pt`` or
    ``ambience_NNNNNN.pt``. Finishes by writing ``metadata.csv`` and
    ``expanded_metadata.csv`` to the same directory.

    Any trailing audio shorter than a full chunk is discarded. Every chunk of a
    recording inherits that recording's label, so a drone recording is assumed to
    contain an audible drone throughout.

    ``file_id`` is assigned by enumeration order over the discovered files and is
    therefore stable only for a fixed set of input files.
    """
    config = load_default_config()
    RAW_DIR = Path(config["paths"]["data_raw"])
    PROCESSED_DIR = Path(config["paths"]["data_processed"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    spectrogram_transform = SpectrogramTransform(config)

    CHUNK_DURATION = config['data']['duration']
    audio_files = list(RAW_DIR.rglob("*.wav"))
    print(f"Found {len(audio_files)} raw audio files. Starting processing...")
    drone_spec_number = 0
    no_drone_spec_number = 0
    metadata_rows = []
    expanded_metadata_rows = []
    for file_id, audio_path in enumerate(tqdm(audio_files)):
        meta = parse_filename_metadata(audio_path.name)
        if not meta:
            continue
        waveform, sr = torchaudio.load(audio_path)
        if (meta["sr"] != 0 and sr != meta["sr"]):
            print("whoops, sr incorrect!!!!")
        chunk_samples = int(CHUNK_DURATION * sr)
        total_samples = waveform.shape[1]
        num_chunks = total_samples / chunk_samples
        for i in range (int(num_chunks)):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk_waveform = waveform[:, start:end]
            spectrogram = spectrogram_transform(chunk_waveform)
            if (meta['is_drone']):
                prefix = 'drone'
                num = drone_spec_number
                drone_spec_number += 1
            else:
                prefix = 'ambience'
                num = no_drone_spec_number
                no_drone_spec_number += 1
            out_filename = f"{prefix}_{num:06d}.pt"
            out_path = PROCESSED_DIR / out_filename
            torch.save(spectrogram, out_path)
            chunk_meta = {}
            chunk_meta["filename"] = out_filename
            chunk_meta["file_id"] = file_id
            chunk_meta["target"] = 1 if meta["is_drone"] else 0
            metadata_rows.append(chunk_meta)
            expanded_chunk_meta = meta.copy()
            expanded_chunk_meta["filename"] = out_filename
            expanded_metadata_rows.append(expanded_chunk_meta)
    if not metadata_rows:
        print("No chunks were produced — check that data/raw/ contains WAV files "
              "matching the expected filename format.")
        return

    df = pd.DataFrame(metadata_rows)
    cols = ['filename', 'target'] + [c for c in df.columns if c not in ['filename', 'target']]
    df = df[cols]

    expanded_df = pd.DataFrame(expanded_metadata_rows)
    cols = ['filename', 'is_drone'] + [c for c in expanded_df.columns if c not in ['filename', 'is_drone']]
    expanded_df = expanded_df[cols]

    csv_path = PROCESSED_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False)
    expanded_csv_path = PROCESSED_DIR / "expanded_metadata.csv"
    expanded_df.to_csv(expanded_csv_path, index=False)

    print(f"Processing complete. Metadata saved to {csv_path}")
    print(f"Total processed chunks: {len(df)}")




if __name__ == "__main__":
    process_data()
