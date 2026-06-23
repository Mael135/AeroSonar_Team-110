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
    # Convert dB to Amplitude (Power 1.0 is Amplitude, 2.0 is Power)
    # Most SFM formulas use Power Spectrograms.
    amp_spec = F.DB_to_amplitude(spectrogram, ref=1.0, power=2.0)
    
    eps = 1e-10
    # SFM calculation
    log_spec = torch.log(amp_spec + eps)
    g_mean = torch.exp(torch.mean(log_spec, dim=0))
    a_mean = torch.mean(amp_spec, dim=0)
    
    sfm_per_frame = g_mean / (a_mean + eps)
    
    # Use the 10th percentile or Min rather than Mean. 
    # Even a far drone will only be 'tonal' in some frames.
    val_to_check = torch.quantile(sfm_per_frame, 0.1).item()
    print(val_to_check)
    return 1 if val_to_check < threshold else 0



def mask_spec_prominence(spectrogram: torch.Tensor, db_threshold: float = 30) -> int:
    # This works well directly on dB Mel-spectrograms
    # Calculate the max intensity bin vs the median intensity bin across the 2s chunk
    max_val = torch.max(spectrogram)
    median_val = torch.median(spectrogram)
    
    # Difference in dB acts as a Signal-to-Noise Ratio proxy
    diff = max_val - median_val
    print(diff)
    return 1 if diff.item() > db_threshold else 0



def mask_spec_variance(spectrogram: torch.Tensor, var_threshold: float = 100) -> int:
    # Calculate the variance of the energy across the time dimension
    time_energy = torch.mean(spectrogram, dim=0) # Average over Mel-bins per time frame
    variance = torch.var(time_energy)
    print(variance.item())
    # If variance is too high, it might be transient noise (speech, impact)
    # If variance is extremely low, it might be silence or a steady drone
    return 1 if variance.item() < var_threshold else 0



def parse_filename_metadata(filename: str):
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