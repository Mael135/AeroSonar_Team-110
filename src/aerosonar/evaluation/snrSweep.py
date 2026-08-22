"""Task 9: signal-to-noise ratio sensitivity.

Takes drone clips the model detects confidently, buries them in progressively louder
noise, and measures how the confidence decays. The quantity of interest is the operating
floor: the signal-to-noise ratio at which the detection rate falls through one half.

Noise is mixed in the waveform domain, before the mel transform, so it undergoes exactly
the processing that real noise would. Two noise colours are used. White noise has a flat
spectrum and is the standard reference. Pink noise falls as 1/f, which is the shape wind,
traffic and general outdoor ambience exhibit, and concentrates its energy in the low bands
where rotor harmonics lie.

The sweep is accompanied by a negative control of noise containing no drone at all.
Without it, a curve that stays above threshold at very low signal-to-noise ratio reads as
noise robustness when it may instead reflect a model biased toward firing.

RMS loudness normalisation scales signal and noise together and so leaves the ratio
unchanged, meaning the sweep measures noise tolerance rather than an artefact of that
normalisation.

Run from the repository root::

    python -m aerosonar.evaluation.snrSweep
"""
from pathlib import Path

import numpy as np
import torch
import torchaudio

from aerosonar.config import load_default_config
from aerosonar.evaluation.common import (graphs_dir, load_deployed_threshold,
                                         load_trained_model, reports_dir, resolve_device,
                                         section, verdict, write_csv, write_json)
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.utils.plotting import COLORS, finish, new_figure
from aerosonar.utils.seeding import seed_everything

#: Detection rate whose downward crossing defines the operating floor.
DETECTION_FLOOR = 0.5


def pink_noise(n, rng):
    """Generate pink (1/f) noise by spectrally shaping white noise.

    Args:
        n: Number of samples.
        rng: Seeded NumPy generator.

    Returns:
        np.ndarray: Noise of unit standard deviation.
    """
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(n)
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    shaped = np.fft.irfft(spectrum * scale, n=n)
    return shaped / (np.std(shaped) or 1.0)


def mix_at_snr(signal, noise, snr_db):
    """Mix noise into a signal at an exact signal-to-noise ratio.

    Args:
        signal: The clean signal.
        noise: Noise to add, rescaled to hit the target ratio.
        snr_db: Target signal-to-noise ratio in decibels.

    Returns:
        np.ndarray: The mixture.
    """
    signal_power = float(np.mean(signal ** 2))
    noise_power = float(np.mean(noise ** 2)) or 1e-12
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    return signal + noise * np.sqrt(target_noise_power / noise_power)


def select_clips(config, model, transform, device, n_clips):
    """Select the drone clips the model scores most confidently.

    The sweep must begin from clips the model actually detects. Starting from clips it
    already misses would produce a flat, uninformative curve, since confidence would be
    low at every signal-to-noise ratio for reasons unrelated to the added noise.

    Args:
        config: Project configuration.
        model: Trained model in evaluation mode.
        transform: Feature extractor, already placed on ``device``.
        device: Compute device.
        n_clips: Number of clips to return.

    Returns:
        tuple: ``(clips, n_scanned)`` where ``clips`` is a list of
        ``(probability, filename, chunk_index, samples)`` ordered by descending
        confidence, and ``n_scanned`` is the number of candidates examined. Returns
        ``([], None)`` if no drone recordings exist.
    """
    raw_dir = Path(config["paths"]["data_raw"])
    window = int(config["data"]["sample_rate"] * config["data"]["duration"])
    drone_files = sorted(raw_dir.rglob("*__drone__*.wav"))
    if not drone_files:
        return [], None

    candidates = []
    for wav_path in drone_files:
        waveform, _ = torchaudio.load(wav_path)
        audio = waveform.mean(dim=0).numpy().astype(np.float32)
        n_chunks = min(60, audio.shape[0] // window)
        for index in range(n_chunks):
            clip = audio[index * window:(index + 1) * window]
            with torch.no_grad():
                tensor = torch.from_numpy(clip).float().to(device).unsqueeze(0)
                spec = transform(tensor).unsqueeze(0)
                probability = torch.softmax(model(spec), dim=1)[0, 1].item()
            candidates.append((probability, wav_path.name, index, clip))

    candidates.sort(key=lambda c: -c[0])
    return candidates[:n_clips], len(candidates)


def run(config=None):
    """Sweep signal-to-noise ratios and run the noise-only control.

    Writes ``snr_sweep.csv``, ``snr_sweep.json`` and ``snr_sensitivity.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, the per-condition sweep, the clean
        reference confidence, the noise-only control, the operating floor per noise
        colour, and the individual check outcomes.
    """
    config = config or load_default_config()
    eval_config = config["evaluation"]
    device = resolve_device(config)
    section("TASK 9 — SIGNAL-TO-NOISE RATIO SENSITIVITY")
    seed_everything(config["data"].get("seed", 42))

    snr_levels = eval_config["snr_db_levels"]
    n_seeds = eval_config["snr_noise_seeds"]
    n_clips = eval_config["snr_num_clips"]
    threshold = load_deployed_threshold(config)

    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(device)
    transform.amplitude_to_db = transform.amplitude_to_db.to(device)
    model = load_trained_model(config, device)

    clips, n_scanned = select_clips(config, model, transform, device, n_clips)
    if not clips:
        print("  No drone recordings found; skipping.")
        return {"task": 9, "name": "SNR sensitivity", "status": "SKIP",
                "reason": "no drone recordings"}

    print(f"Scanned {n_scanned} drone chunks; using the {len(clips)} highest-confidence "
          f"(clean p(drone) {clips[-1][0]:.3f} to {clips[0][0]:.3f})")
    print(f"Deployed threshold {threshold:.2f} | {n_seeds} noise seeds per point | "
          f"SNR levels {snr_levels} dB\n")

    rows = []
    clean_probs = [c[0] for c in clips]
    for noise_index, noise_kind in enumerate(("white", "pink")):
        for snr_index, snr_db in enumerate(snr_levels):
            probabilities = []
            for clip_index, (_, filename, chunk_index, clip) in enumerate(clips):
                for seed in range(n_seeds):
                    # Seeded per condition so the sweep is reproducible. Indices are
                    # used rather than the decibel value because the levels go negative
                    # and default_rng rejects negative seeds.
                    rng = np.random.default_rng([noise_index, snr_index, clip_index, seed])
                    noise = (rng.standard_normal(clip.shape[0]) if noise_kind == "white"
                             else pink_noise(clip.shape[0], rng))
                    mixed = mix_at_snr(clip, noise.astype(np.float32), snr_db)
                    with torch.no_grad():
                        tensor = torch.from_numpy(mixed).float().to(device).unsqueeze(0)
                        spec = transform(tensor).unsqueeze(0)
                        probabilities.append(
                            torch.softmax(model(spec), dim=1)[0, 1].item())
            probabilities = np.array(probabilities)
            rows.append({
                "noise": noise_kind,
                "snr_db": snr_db,
                "n": len(probabilities),
                "mean_confidence": round(float(probabilities.mean()), 6),
                "std_confidence": round(float(probabilities.std()), 6),
                "min_confidence": round(float(probabilities.min()), 6),
                "max_confidence": round(float(probabilities.max()), 6),
                "detection_rate": round(float((probabilities > threshold).mean()), 6),
            })
            row = rows[-1]
            print(f"  {noise_kind:5s} {snr_db:+4d} dB : p(drone) = "
                  f"{row['mean_confidence']:.4f} +/- {row['std_confidence']:.4f} | "
                  f"detection rate {row['detection_rate']:6.1%}")

    clean_mean = float(np.mean(clean_probs))
    print(f"\n  clean (no added noise) : p(drone) = {clean_mean:.4f} | "
          f"detection rate {float(np.mean([p > threshold for p in clean_probs])):.1%}")

    # Negative control: noise containing no drone. If these score above threshold too,
    # the sweep above is measuring bias rather than sensitivity.
    control = {}
    for noise_index, noise_kind in enumerate(("white", "pink")):
        probabilities = []
        for repeat in range(n_seeds * len(clips)):
            rng = np.random.default_rng([99, noise_index, repeat])
            n = clips[0][3].shape[0]
            noise = (rng.standard_normal(n) if noise_kind == "white" else pink_noise(n, rng))
            with torch.no_grad():
                tensor = torch.from_numpy(noise.astype(np.float32)).float().to(device).unsqueeze(0)
                spec = transform(tensor).unsqueeze(0)
                probabilities.append(torch.softmax(model(spec), dim=1)[0, 1].item())
        probabilities = np.array(probabilities)
        control[noise_kind] = {
            "n": len(probabilities),
            "mean_confidence": round(float(probabilities.mean()), 6),
            "std_confidence": round(float(probabilities.std()), 6),
            "false_alarm_rate": round(float((probabilities > threshold).mean()), 6),
        }
        entry = control[noise_kind]
        print(f"  {noise_kind:5s} NOISE ONLY (no drone) : p(drone) = "
              f"{entry['mean_confidence']:.4f} +/- {entry['std_confidence']:.4f} | "
              f"false-alarm rate {entry['false_alarm_rate']:6.1%}")
    noise_only_rejected = all(c["false_alarm_rate"] < 0.5 for c in control.values())
    if not noise_only_rejected:
        print("    -> The model fires on pure noise containing no drone. The confidence "
              "retained at low SNR above is therefore a positive bias, not noise robustness.")

    floors = {}
    for noise_kind in ("white", "pink"):
        series = sorted((r for r in rows if r["noise"] == noise_kind),
                        key=lambda r: -r["snr_db"])
        floors[noise_kind] = _crossing(series)
        label = (f"{floors[noise_kind]:.1f} dB" if floors[noise_kind] is not None
                 else f"never falls below {DETECTION_FLOOR:.0%} over the swept range")
        print(f"  operating floor ({noise_kind:5s}) : {label}")

    checks = {
        "clean_clips_are_detected": clean_mean > threshold,
        "confidence_degrades_with_noise": all(
            rows[0]["mean_confidence"] >= r["mean_confidence"]
            for r in rows if r["noise"] == "white" and r["snr_db"] == min(snr_levels)
        ),
        "operating_floor_within_swept_range": any(v is not None for v in floors.values()),
        "noise_only_rejected": noise_only_rejected,
    }
    status = verdict(all(checks.values()))
    print()
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 9: {status}")

    result = {
        "task": 9, "name": "SNR sensitivity", "status": status,
        "threshold": threshold,
        "snr_db_levels": snr_levels,
        "noise_seeds_per_point": n_seeds,
        "clips_used": [{"file": c[1], "chunk": c[2], "clean_probability": c[0]} for c in clips],
        "clean_mean_confidence": clean_mean,
        "operating_floor_db": floors,
        "detection_floor": DETECTION_FLOOR,
        "noise_only_control": control,
        "sweep": rows,
        "checks": checks,
    }
    write_csv(rows, reports_dir(config) / "snr_sweep.csv")
    result["figure"] = _plot(rows, floors, threshold, clean_mean, control, config)
    write_json(result, reports_dir(config) / "snr_sweep.json")
    return result


def _crossing(series):
    """Interpolate the signal-to-noise ratio at which detection falls through the floor.

    Args:
        series: Sweep records for one noise colour, ordered from the highest ratio
            downward, so the crossing is the first point below
            :data:`DETECTION_FLOOR`.

    Returns:
        float | None: The interpolated ratio in decibels, or None if the detection rate
        never falls through the floor within the swept range.
    """
    for previous, current in zip(series, series[1:]):
        if previous["detection_rate"] >= DETECTION_FLOOR > current["detection_rate"]:
            span = previous["detection_rate"] - current["detection_rate"]
            if span == 0:
                return current["snr_db"]
            fraction = (previous["detection_rate"] - DETECTION_FLOOR) / span
            return previous["snr_db"] + fraction * (current["snr_db"] - previous["snr_db"])
    return None


def _plot(rows, floors, threshold, clean_mean, control, config):
    """Plot confidence against signal-to-noise ratio for both noise colours.

    The noise-only control level is drawn for each colour, since the curves converge on
    it, and the detection threshold and clean-clip confidence are marked for reference.

    Args:
        rows: Sweep records.
        floors: Operating floor per noise colour, where one was found.
        threshold: Detection threshold.
        clean_mean: Mean confidence on the unmodified clips.
        control: Noise-only control measurements.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    fig = new_figure(figsize=(8, 5))
    ax = fig.gca()
    for noise_kind, color in (("white", COLORS["train"]), ("pink", COLORS["val"])):
        series = sorted((r for r in rows if r["noise"] == noise_kind),
                        key=lambda r: r["snr_db"])
        snr = [r["snr_db"] for r in series]
        mean = np.array([r["mean_confidence"] for r in series])
        std = np.array([r["std_confidence"] for r in series])
        ax.plot(snr, mean, color=color, linewidth=2, marker="o", label=f"{noise_kind} noise")
        ax.fill_between(snr, mean - std, mean + std, color=color, alpha=0.18)
        floor = floors.get(noise_kind)
        if floor is not None:
            ax.axvline(floor, color=color, linestyle=":", linewidth=1.5,
                       label=f"{noise_kind} operating floor ({floor:.1f} dB)")
        if control and noise_kind in control:
            ax.axhline(control[noise_kind]["mean_confidence"], color=color, linestyle=":",
                       linewidth=1.5, alpha=0.9,
                       label=f"{noise_kind} noise only, no drone "
                             f"({control[noise_kind]['mean_confidence']:.2f})")

    ax.axhline(threshold, color=COLORS["f1"], linestyle="--", linewidth=2,
               label=f"Detection threshold ({threshold:.2f})")
    ax.axhline(clean_mean, color=COLORS["reference"], linestyle="-.", linewidth=1,
               label=f"Clean-clip confidence ({clean_mean:.2f})")
    ax.set_title("Model confidence vs signal-to-noise ratio")
    ax.set_xlabel("SNR (dB) — lower is noisier")
    ax.set_ylabel("Mean p(drone), shaded ±1 sd")
    ax.set_ylim(0, 1.02)
    ax.invert_xaxis()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    return finish(graphs_dir(config) / "snr_sensitivity.png")


if __name__ == "__main__":
    run()
