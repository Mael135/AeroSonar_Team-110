# Diagnostic sweep over the raw audio, processed dataset, and trained model —
# built to surface dataset confounds (location/label shortcuts), label noise
# (drone-labeled chunks with no drone actually audible), and model error
# concentration (which sessions drive precision/recall) without retraining.
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
import yaml

from aerosonar.config import load_default_config
from aerosonar.data.preprocessData import parse_filename_metadata
from aerosonar.models.spectrogramCNN import SpectrogramCNN

REPORT_DIR = Path("reports/audit")


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# 1. Raw audio sanity checks
# ---------------------------------------------------------------------------

def audit_raw_audio(config):
    section("1. RAW AUDIO FILES")
    raw_dir = Path(config["paths"]["data_raw"])
    files = sorted(raw_dir.rglob("*.wav"))
    rows = []
    for f in files:
        meta = parse_filename_metadata(f.name)
        waveform, sr = torchaudio.load(f)
        mono = waveform.mean(dim=0)
        actual_duration = mono.shape[0] / sr
        peak = mono.abs().max().item()
        rms = mono.pow(2).mean().sqrt().item()
        dbfs = 20 * np.log10(rms + 1e-12)
        silence_frac = (mono.abs() < 0.01).float().mean().item()
        rows.append({
            "file": f.name, "location": meta["location"], "is_drone": meta["is_drone"],
            "gain": meta["gain"], "claimed_sr": meta["sr"], "actual_sr": sr,
            "sr_mismatch": sr != meta["sr"],
            "claimed_duration_s": meta["duration"], "actual_duration_s": round(actual_duration, 1),
            "duration_mismatch": abs(actual_duration - meta["duration"]) > 2,
            "peak_amp": round(peak, 4), "clipping": peak >= 0.99,
            "rms_dbfs": round(dbfs, 1), "silence_frac": round(silence_frac, 3),
        })
    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "raw_audio_audit.csv", index=False)

    print(f"Checked {len(df)} raw files.")
    if df["sr_mismatch"].any():
        print(f"\n  WARNING: {df['sr_mismatch'].sum()} files have a sample-rate mismatch vs filename metadata:")
        print(df[df["sr_mismatch"]][["file", "claimed_sr", "actual_sr"]].to_string(index=False))
    if df["duration_mismatch"].any():
        print(f"\n  WARNING: {df['duration_mismatch'].sum()} files differ >2s from their filename-claimed duration:")
        print(df[df["duration_mismatch"]][["file", "claimed_duration_s", "actual_duration_s"]].to_string(index=False))
    if df["clipping"].any():
        print(f"\n  WARNING: {df['clipping'].sum()} files show clipping (peak amplitude >= 0.99):")
        print(df[df["clipping"]][["file", "peak_amp"]].to_string(index=False))

    print("\nRMS level (dBFS) by gain setting — a large spread here means the gain knob really did shift recording level:")
    print(df.groupby("gain")["rms_dbfs"].agg(["mean", "std", "min", "max", "count"]).to_string())
    print("\nSilence fraction by class (file-level, coarse — high values on 'drone' files hint at non-continuous drone presence):")
    print(df.groupby("is_drone")["silence_frac"].agg(["mean", "max", "count"]).to_string())
    print(f"\nFull table saved to {REPORT_DIR / 'raw_audio_audit.csv'}")
    return df


# ---------------------------------------------------------------------------
# 2. Dataset composition / location-label confound check
# ---------------------------------------------------------------------------

def load_joined_metadata(config):
    processed_dir = Path(config["paths"]["data_processed"])
    meta = pd.read_csv(processed_dir / "metadata.csv")
    expanded = pd.read_csv(processed_dir / "expanded_metadata.csv")
    return meta.merge(expanded, on="filename")


def audit_dataset_composition(joined):
    section("2. DATASET COMPOSITION & LOCATION/LABEL CONFOUND")

    n_recordings = joined["file_id"].nunique()
    print(f"{len(joined)} chunks from {n_recordings} unique recordings.")
    print(f"Class balance (chunks): {joined['target'].value_counts().to_dict()}")

    rec_level = joined.drop_duplicates("file_id")[["file_id", "location", "target", "noise_type", "gain"]]
    print(f"\nRecordings per location:\n{rec_level['location'].value_counts().to_string()}")

    print("\nLocation x label crosstab (recording counts) — a location with only one non-zero column is a shortcut risk:")
    ct = pd.crosstab(rec_level["location"], rec_level["target"])
    print(ct.to_string())
    single_class_locations = ct[(ct == 0).any(axis=1)].index.tolist()
    if single_class_locations:
        print(f"\n  WARNING: single-class locations (model can use location as a free label predictor): {single_class_locations}")

    print("\nNoise type x label crosstab (recording counts):")
    print(pd.crosstab(rec_level["noise_type"], rec_level["target"]).to_string())

    print("\nGain x label crosstab (recording counts):")
    print(pd.crosstab(rec_level["gain"], rec_level["target"]).to_string())

    return single_class_locations


# ---------------------------------------------------------------------------
# 3. Train/test split vs location (replicates the seed=42 split in dataset.py)
# ---------------------------------------------------------------------------

def replicate_split(joined, train_part=0.8):
    unique_file_ids = joined["file_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_file_ids)
    train_count = int(train_part * len(unique_file_ids))
    train_ids = set(unique_file_ids[:train_count])
    return joined["file_id"].apply(lambda x: "train" if x in train_ids else "test")


def audit_split_confound(joined):
    section("3. TRAIN/TEST SPLIT vs LOCATION (replicated seed=42 split from dataset.py)")
    joined = joined.copy()
    joined["split"] = replicate_split(joined)
    rec_level = joined.drop_duplicates("file_id")[["file_id", "location", "target", "split"]]
    print(rec_level.groupby(["split", "location"])["target"].agg(["count", "mean"]).to_string())

    test_locations = set(rec_level[rec_level.split == "test"]["location"])
    train_locations = set(rec_level[rec_level.split == "train"]["location"])
    only_in_test = test_locations - train_locations
    only_in_train = train_locations - test_locations
    if only_in_test:
        print(f"\n  WARNING: locations ONLY in test set (test never exercised in training): {only_in_test}")
    if only_in_train:
        print(f"  WARNING: locations ONLY in train set (test never validates generalization here): {only_in_train}")
    return joined


# ---------------------------------------------------------------------------
# 4. Spectrogram level stats vs recording gain (did normalization work?)
# ---------------------------------------------------------------------------

def audit_spectrogram_levels(joined, config):
    section("4. SPECTROGRAM LEVEL STATS (post-normalization) vs RECORDING GAIN")
    processed_dir = Path(config["paths"]["data_processed"])
    rows = []
    for _, row in joined.iterrows():
        spec = torch.load(processed_dir / row["filename"], weights_only=True)
        rows.append({
            "filename": row["filename"], "gain": row["gain"], "target": row["target"],
            "location": row["location"], "mean_db": spec.mean().item(),
            "std_db": spec.std().item(), "max_db": spec.max().item(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "spectrogram_levels.csv", index=False)

    print("Mean spectrogram dB level by gain setting (should now be close across gains if normalization is working):")
    print(df.groupby("gain")["mean_db"].agg(["mean", "std", "count"]).to_string())
    spread = df.groupby("gain")["mean_db"].mean()
    if len(spread) > 1 and (spread.max() - spread.min()) > 3.0:
        print(f"\n  WARNING: mean dB level still differs by {spread.max() - spread.min():.1f} dB across gain settings"
              " — normalization may not be fully equalizing real recordings (check for clipping/noise-floor effects).")

    print("\nMean spectrogram dB level by label:")
    print(df.groupby("target")["mean_db"].agg(["mean", "std", "count"]).to_string())
    return df


# ---------------------------------------------------------------------------
# 5. Chunk-level drone-presence heuristics (catches mislabeled/silent chunks)
# ---------------------------------------------------------------------------

def _sfm_p10(spec_db):
    amp = AF.DB_to_amplitude(spec_db, ref=1.0, power=2.0)
    eps = 1e-10
    log_spec = torch.log(amp + eps)
    g_mean = torch.exp(torch.mean(log_spec, dim=0))
    a_mean = torch.mean(amp, dim=0)
    sfm_per_frame = g_mean / (a_mean + eps)
    return torch.quantile(sfm_per_frame, 0.1).item()


def _prominence_db(spec_db):
    return (spec_db.max() - spec_db.median()).item()


def _time_energy_variance(spec_db):
    time_energy = torch.mean(spec_db, dim=0)
    return torch.var(time_energy).item()


def audit_chunk_presence(joined, config):
    section("5. CHUNK-LEVEL DRONE-PRESENCE HEURISTICS (catches mislabeled/silent chunks)")
    processed_dir = Path(config["paths"]["data_processed"])
    rows = []
    for _, row in joined.iterrows():
        spec = torch.load(processed_dir / row["filename"], weights_only=True).squeeze(0)
        rows.append({
            "filename": row["filename"], "target": row["target"], "location": row["location"],
            "file_id": row["file_id"], "sfm_p10": _sfm_p10(spec),
            "prominence_db": _prominence_db(spec), "time_variance": _time_energy_variance(spec),
        })
    df = pd.DataFrame(rows)

    print("Distribution of 'prominence_db' (max - median dB; higher = a louder, more distinct source stands out) by label:")
    print(df.groupby("target")["prominence_db"].describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]].to_string())

    drone = df[df["target"] == 1]
    cutoff = drone["prominence_db"].quantile(0.10)
    suspects = drone[drone["prominence_db"] <= cutoff]
    df.to_csv(REPORT_DIR / "presence_heuristics.csv", index=False)

    print(f"\nBottom 10% of drone-labeled chunks by prominence (cutoff={cutoff:.1f} dB) — weakest candidates for an audible drone:")
    print(f"  {len(suspects)} chunks, broken down by recording (file_id):")
    print(suspects.groupby("file_id").size().to_string())
    suspects.to_csv(REPORT_DIR / "suspect_mislabeled_drone_chunks.csv", index=False)
    print(f"  Full list saved to {REPORT_DIR / 'suspect_mislabeled_drone_chunks.csv'} — worth spot-listening to a few.")
    return df


# ---------------------------------------------------------------------------
# 6. Trained model — per-chunk predictions & error concentration
# ---------------------------------------------------------------------------

def _confusion(df, pred_col, target_col="target"):
    TP = int(((df[pred_col] == 1) & (df[target_col] == 1)).sum())
    TN = int(((df[pred_col] == 0) & (df[target_col] == 0)).sum())
    FP = int(((df[pred_col] == 1) & (df[target_col] == 0)).sum())
    FN = int(((df[pred_col] == 0) & (df[target_col] == 1)).sum())
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else float("nan")
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "n": total, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


def _fmt(c):
    return (f"n={c['n']:4d}  TP={c['TP']:3d} FP={c['FP']:3d} FN={c['FN']:3d} TN={c['TN']:3d}  "
            f"acc={c['acc']:.3f} prec={c['prec']:.3f} rec={c['rec']:.3f} f1={c['f1']:.3f}")


def audit_model_predictions(joined, config, device, presence_df=None):
    section("6. TRAINED MODEL — PER-CHUNK PREDICTIONS & ERROR CONCENTRATION")
    processed_dir = Path(config["paths"]["data_processed"])
    weights_dir = Path(config["paths"]["weights"])
    weights_path = weights_dir / "CNN_best.pth"
    if not weights_path.exists():
        print(f"  No trained weights found at {weights_path}, skipping model audit.")
        return None

    tuned_threshold = 0.5
    threshold_path = weights_dir / "threshold.yaml"
    if threshold_path.exists():
        tuned_threshold = float(yaml.safe_load(threshold_path.read_text())["detection_threshold"])

    model = SpectrogramCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    joined = joined.copy()
    joined["split"] = replicate_split(joined)

    probs = []
    batch, BATCH_SIZE = [], 128
    with torch.no_grad():
        for _, row in joined.iterrows():
            batch.append(torch.load(processed_dir / row["filename"], weights_only=True))
            if len(batch) >= BATCH_SIZE:
                x = torch.stack(batch).to(device).float()
                probs.extend(torch.softmax(model(x), dim=1)[:, 1].cpu().tolist())
                batch = []
        if batch:
            x = torch.stack(batch).to(device).float()
            probs.extend(torch.softmax(model(x), dim=1)[:, 1].cpu().tolist())

    joined["prob"] = probs
    joined["pred_05"] = (joined["prob"] > 0.5).astype(int)
    joined["pred_tuned"] = (joined["prob"] > tuned_threshold).astype(int)
    joined.to_csv(REPORT_DIR / "chunk_predictions.csv", index=False)

    test_df = joined[joined.split == "test"]
    print(f"Test split: {len(test_df)} chunks from {test_df['file_id'].nunique()} recordings.")
    print(f"  @ threshold=0.50          : {_fmt(_confusion(test_df, 'pred_05'))}")
    print(f"  @ tuned threshold={tuned_threshold:.2f}     : {_fmt(_confusion(test_df, 'pred_tuned'))}")

    print("\nPer-location breakdown on test split (tuned threshold) — look for locations driving all the errors or all the precision:")
    for loc, g in test_df.groupby("location"):
        print(f"  {loc:25s} {_fmt(_confusion(g, 'pred_tuned'))}")

    print("\nPredicted drone-probability distribution (test split) by true label:")
    print(test_df.groupby("target")["prob"].describe()[["count", "mean", "std", "min", "50%", "max"]].to_string())

    false_negatives = test_df[(test_df["target"] == 1) & (test_df["pred_tuned"] == 0)]
    if len(false_negatives):
        print(f"\nFalse negatives ({len(false_negatives)}) by location/recording:")
        print(false_negatives.groupby(["location", "file_id"]).size().to_string())

        if presence_df is not None:
            merged = false_negatives.merge(presence_df[["filename", "prominence_db"]], on="filename", how="left")
            low_prominence_fn = merged[merged["prominence_db"] <= presence_df[presence_df.target == 1]["prominence_db"].median()]
            print(f"\n  {len(low_prominence_fn)}/{len(false_negatives)} false negatives also have below-median drone-prominence"
                  " — likely genuinely weak/absent drone signal rather than a model failure.")

    return joined


# ---------------------------------------------------------------------------

def main():
    config = load_default_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    section("CONFIG")
    print(yaml.dump(config))

    audit_raw_audio(config)
    joined = load_joined_metadata(config)
    single_class_locations = audit_dataset_composition(joined)
    audit_split_confound(joined)
    audit_spectrogram_levels(joined, config)
    presence_df = audit_chunk_presence(joined, config)
    audit_model_predictions(joined, config, device, presence_df=presence_df)

    section("SUMMARY — RED FLAGS TO FOLLOW UP")
    if single_class_locations:
        print(f"- Single-class locations exist: {single_class_locations} — model may be using location as a label shortcut.")
    print(f"- See {REPORT_DIR}/ for full per-chunk CSVs: chunk_predictions.csv, presence_heuristics.csv,")
    print("  suspect_mislabeled_drone_chunks.csv, spectrogram_levels.csv, raw_audio_audit.csv.")


if __name__ == "__main__":
    main()
