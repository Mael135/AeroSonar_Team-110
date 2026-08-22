"""Task 4: pipeline continuity test.

Streams long continuous audio through the deployed rolling-buffer inference loop and
tracks resident memory, verifying that real-time spectrogram generation does not leak.

A leak, whether a growing queue, an accumulating list or a retained autograd graph,
appears in no accuracy metric and in no short run. It appears hours into a deployment as
an out-of-memory termination.

Two sources are streamed: silence, which is the pathological case for the loudness
normaliser because it applies the full gain cap to a near-zero RMS, and real recorded
audio.

The loop drives :func:`aerosonar.inference.streaming.push_frame`, the same buffer update
the live detector uses, so a regression there is caught here. ``sounddevice`` is not
imported, allowing the check to run on hosts with no audio device.

Run from the repository root::

    python -m aerosonar.evaluation.continuityTest
"""
import gc
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from aerosonar.config import load_default_config
from aerosonar.evaluation.common import (load_trained_model, reports_dir, resolve_device,
                                         section, verdict, write_csv, write_json)
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.inference.streaming import push_frame
from aerosonar.utils.plotting import COLORS, finish, new_figure

#: Memory drift above this rate counts as a leak. A steady 1 MB/min would consume
#: roughly 1.4 GB per day.
MAX_DRIFT_MB_PER_MIN = 1.0

#: Leading fraction of samples excluded from the drift fit. Allocator warm-up and lazy
#: CUDA context creation are not leaks.
BURN_IN_FRACTION = 0.10

#: Frames between resident-memory samples.
SAMPLE_EVERY = 50


def read_rss_mb():
    """Read the process's current resident set size.

    Reads ``/proc/self/status``, which avoids a dependency on ``psutil``.

    Returns:
        float: Resident set size in megabytes, or NaN where ``/proc`` is unavailable.
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _source_audio(config, kind, total_samples):
    """Produce the audio to stream.

    Args:
        config: Project configuration.
        kind: Either ``"silence"`` or ``"recorded"``.
        total_samples: Number of samples required.

    Returns:
        np.ndarray | None: The audio, or None if ``recorded`` was requested and no WAV
        files are available. Recorded audio is tiled to reach the requested length.
    """
    if kind == "silence":
        return np.zeros(total_samples, dtype=np.float32)
    raw_dir = Path(config["paths"]["data_raw"])
    wav_files = sorted(raw_dir.rglob("*.wav"))
    if not wav_files:
        return None
    waveform, _ = torchaudio.load(wav_files[0])
    audio = waveform.mean(dim=0).numpy().astype(np.float32)
    if audio.shape[0] < total_samples:
        audio = np.tile(audio, int(np.ceil(total_samples / audio.shape[0])))
    return audio[:total_samples]


def stream(config, model, transform, device, kind, minutes, block_samples, window_size):
    """Stream audio through the inference loop, sampling memory as it runs.

    Advances the rolling window one block at a time and runs inference once the window
    has filled, mirroring the live detector. Memory is sampled every
    :data:`SAMPLE_EVERY` frames, and drift is estimated by a linear fit over the
    steady-state region following the burn-in.

    Args:
        config: Project configuration.
        model: The trained model, in evaluation mode.
        transform: Feature extractor, already placed on ``device``.
        device: Compute device.
        kind: Audio source, ``"silence"`` or ``"recorded"``.
        minutes: Duration to stream.
        block_samples: Samples per simulated stream callback.
        window_size: Length of the rolling analysis window in samples.

    Returns:
        tuple | None: ``(summary, samples)`` where ``summary`` holds the frame and
        inference counts, memory figures, drift rate and mean drone probability, and
        ``samples`` is the per-sample memory trace. None if no source audio exists.
    """
    sr = config["data"]["sample_rate"]
    total_samples = int(minutes * 60 * sr)
    audio = _source_audio(config, kind, total_samples)
    if audio is None:
        return None

    n_frames = total_samples // block_samples
    buffer = np.zeros(window_size, dtype=np.float32)
    samples = []
    probabilities = []

    gc.collect()
    tracemalloc.start()
    baseline_rss = read_rss_mb()

    for frame in range(n_frames):
        block = audio[frame * block_samples:(frame + 1) * block_samples]
        push_frame(buffer, block)

        # Warm-up: the buffer has not filled once yet, mirroring inference.py.
        if (frame + 1) * block_samples < window_size:
            continue

        with torch.no_grad():
            tensor = torch.from_numpy(buffer).float().to(device).unsqueeze(0)
            spec = transform(tensor).unsqueeze(0)
            probabilities.append(torch.softmax(model(spec), dim=1)[0, 1].item())

        if frame % SAMPLE_EVERY == 0:
            samples.append({
                "source": kind,
                "frame": frame,
                "stream_seconds": round(frame * block_samples / sr, 2),
                "rss_mb": round(read_rss_mb(), 3),
                "tracemalloc_mb": round(tracemalloc.get_traced_memory()[0] / 1024**2, 3),
            })

    traced_peak_mb = tracemalloc.get_traced_memory()[1] / 1024**2
    tracemalloc.stop()

    burn_in = int(len(samples) * BURN_IN_FRACTION)
    steady = samples[burn_in:]
    minutes_axis = np.array([s["stream_seconds"] / 60.0 for s in steady])
    rss_axis = np.array([s["rss_mb"] for s in steady])
    drift = float(np.polyfit(minutes_axis, rss_axis, 1)[0]) if len(steady) > 2 else 0.0

    summary = {
        "source": kind,
        "minutes_streamed": minutes,
        "frames": n_frames,
        "inferences": len(probabilities),
        "baseline_rss_mb": round(baseline_rss, 2),
        "steady_state_rss_mb": round(float(rss_axis.mean()), 2),
        "final_rss_mb": round(samples[-1]["rss_mb"], 2),
        "rss_growth_mb": round(samples[-1]["rss_mb"] - rss_axis[0], 2),
        "drift_mb_per_min": round(drift, 4),
        "tracemalloc_peak_mb": round(traced_peak_mb, 2),
        "mean_drone_probability": round(float(np.mean(probabilities)), 4) if probabilities else None,
        "flat_memory": abs(drift) < MAX_DRIFT_MB_PER_MIN,
    }
    print(f"  {kind:9s}: {n_frames:6d} frames, {len(probabilities):6d} inferences | "
          f"RSS {rss_axis[0]:7.1f} -> {samples[-1]['rss_mb']:7.1f} MB | "
          f"drift {drift:+.3f} MB/min | mean p(drone)={summary['mean_drone_probability']}")
    return summary, samples


def run(config=None):
    """Stream both audio sources and assess memory stability.

    Writes ``continuity_memory.csv``, ``continuity_test.json`` and
    ``continuity_memory.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, per-stream summaries, any exception
        raised mid-stream, and the individual check outcomes.
    """
    config = config or load_default_config()
    eval_config = config["evaluation"]
    device = resolve_device(config)
    section("TASK 4 — PIPELINE CONTINUITY TEST (memory stability under a long stream)")

    sr = config["data"]["sample_rate"]
    window_size = int(sr * config["data"]["duration"])
    block_samples = max(1, int(sr * eval_config["continuity_block_ms"] / 1000.0))
    minutes = eval_config["continuity_minutes"]

    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(device)
    transform.amplitude_to_db = transform.amplitude_to_db.to(device)
    model = load_trained_model(config, device)

    print(f"Streaming {minutes} min per source | window {window_size} samples "
          f"({config['data']['duration']}s) | block {block_samples} samples "
          f"({eval_config['continuity_block_ms']} ms) | device {device}")
    print(f"Leak threshold: {MAX_DRIFT_MB_PER_MIN} MB/min of RSS drift after "
          f"{int(BURN_IN_FRACTION * 100)}% burn-in\n")

    summaries, all_samples, crashed = [], [], None
    try:
        for kind in ("silence", "recorded"):
            outcome = stream(config, model, transform, device, kind, minutes,
                             block_samples, window_size)
            if outcome is None:
                print(f"  {kind}: no source audio available, skipped")
                continue
            summary, samples = outcome
            summaries.append(summary)
            all_samples.extend(samples)
    except Exception as exc:
        # A crash part-way through a long stream is itself the finding, so it is
        # recorded rather than propagated.
        crashed = f"{type(exc).__name__}: {exc}"
        print(f"  STREAM CRASHED: {crashed}")

    checks = {
        "no_crash_over_long_stream": crashed is None,
        "completed_all_sources": len(summaries) == 2,
        f"rss_drift_under_{MAX_DRIFT_MB_PER_MIN}mb_per_min": bool(
            summaries and all(s["flat_memory"] for s in summaries)
        ),
    }
    status = verdict(all(checks.values()))
    print()
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 4: {status}")

    result = {
        "task": 4, "name": "Pipeline continuity test", "status": status,
        "minutes_per_source": minutes,
        "block_ms": eval_config["continuity_block_ms"],
        "window_samples": window_size,
        "device": str(device),
        "max_drift_mb_per_min": MAX_DRIFT_MB_PER_MIN,
        "exception": crashed,
        "streams": summaries,
        "checks": checks,
    }
    if all_samples:
        write_csv(all_samples, reports_dir(config) / "continuity_memory.csv")
        result["figure"] = _plot(all_samples, summaries, config)
    write_json(result, reports_dir(config) / "continuity_test.json")
    return result


def _plot(samples, summaries, config):
    """Plot resident memory against stream time for each source.

    Args:
        samples: Combined memory trace from every stream.
        summaries: Per-stream summaries, read for the drift rates shown in the legend.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    new_figure()
    for kind, color in (("silence", COLORS["train"]), ("recorded", COLORS["val"])):
        series = [s for s in samples if s["source"] == kind]
        if not series:
            continue
        drift = next((s["drift_mb_per_min"] for s in summaries if s["source"] == kind), 0.0)
        plt.plot([s["stream_seconds"] / 60.0 for s in series], [s["rss_mb"] for s in series],
                 color=color, linewidth=2, label=f"{kind} ({drift:+.3f} MB/min)")
    plt.title("Resident memory during continuous streaming")
    plt.xlabel("Stream time (minutes)")
    plt.ylabel("RSS (MB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return finish(Path(config["evaluation"]["graphs_dir"]) / "continuity_memory.png")


if __name__ == "__main__":
    run()
