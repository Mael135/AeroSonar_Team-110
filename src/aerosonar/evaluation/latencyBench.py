"""Task 7: latency and bounding measurement.

Times the complete inference path exactly as the live detector runs it: a raw array of
one window of audio in, a single drone-probability scalar out, including the mel
transform and the device transfer. Excluding any of those would understate the real cost.

The pass criterion is the frame budget given by ``evaluation.latency_budget_ms``, judged
on the 99th percentile rather than the mean. A real-time stage is bounded by its worst
frames, so a mean inside budget accompanied by a tail outside it still drops frames.

Both CPU and CUDA are measured. CUDA is the deployment path where available; CPU is the
worst case for a host without a usable GPU, and is the figure to quote when the
deployment target is not fixed.

CUDA timings synchronise before each stop. Kernel launches are asynchronous, so an
unsynchronised timer measures how quickly work can be queued rather than how quickly the
device completes it.

Run from the repository root::

    python -m aerosonar.evaluation.latencyBench
"""
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from aerosonar.config import load_default_config
from aerosonar.evaluation.common import (frame_geometry, graphs_dir, load_trained_model,
                                         reports_dir, resolve_device, section, verdict,
                                         write_csv, write_json)
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.utils.plotting import COLORS, finish, new_figure
from aerosonar.utils.seeding import seed_everything


def _sync(device):
    """Block until queued device work has completed.

    Args:
        device: Device to synchronise. A no-op for CPU.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def _summarise(samples_ms):
    """Reduce a set of timings to summary statistics.

    Args:
        samples_ms: Individual measurements in milliseconds.

    Returns:
        dict: Run count, mean, median, minimum, maximum, standard deviation, and the
        95th and 99th percentiles.
    """
    ordered = sorted(samples_ms)
    return {
        "runs": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
    }


def benchmark(config, device, runs, warmup):
    """Time the inference path on one device.

    Measures the end-to-end path and, separately, the mel transform and the model
    forward pass, so their contributions can be reported individually. A fixed
    pseudo-random input is used so every device performs identical work and the
    comparison is not confounded by differing input content.

    Args:
        config: Project configuration.
        device: Device to benchmark.
        runs: Number of timed iterations.
        warmup: Untimed iterations run first, to exclude lazy initialisation and
            allocator warm-up.

    Returns:
        dict: Device identification, summary statistics for each of the three timed
        stages, and the raw end-to-end samples.
    """
    window = int(config["data"]["sample_rate"] * config["data"]["duration"])
    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(device)
    transform.amplitude_to_db = transform.amplitude_to_db.to(device)
    model = load_trained_model(config, device)

    rng = np.random.default_rng(config["data"].get("seed", 42))
    audio = (rng.standard_normal(window) * 0.1).astype(np.float32)

    def infer():
        """Run one full inference: array to tensor, mel transform, model, scalar.

        Returns:
            float: The drone probability.
        """
        with torch.no_grad():
            tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0)
            spec = transform(tensor).unsqueeze(0)
            return torch.softmax(model(spec), dim=1)[0, 1].item()

    for _ in range(warmup):
        infer()
    _sync(device)

    end_to_end, transform_only, model_only = [], [], []
    for _ in range(runs):
        start = time.perf_counter_ns()
        infer()
        _sync(device)
        end_to_end.append((time.perf_counter_ns() - start) / 1e6)

        with torch.no_grad():
            start = time.perf_counter_ns()
            tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0)
            spec = transform(tensor).unsqueeze(0)
            _sync(device)
            transform_only.append((time.perf_counter_ns() - start) / 1e6)

            start = time.perf_counter_ns()
            torch.softmax(model(spec), dim=1)[0, 1].item()
            _sync(device)
            model_only.append((time.perf_counter_ns() - start) / 1e6)

    return {
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0) if device.type == "cuda"
                        else "CPU"),
        "end_to_end": _summarise(end_to_end),
        "transform_only": _summarise(transform_only),
        "model_only": _summarise(model_only),
        "samples_ms": end_to_end,
    }


def run(config=None):
    """Benchmark inference latency against the frame budget.

    Writes ``latency.csv``, ``latency.json`` and ``latency_histogram.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, the frame geometry, per-device timing
        summaries with headroom against the budget, the analysis hop and the clip
        cadence, and the individual check outcomes.
    """
    config = config or load_default_config()
    eval_config = config["evaluation"]
    budget = eval_config["latency_budget_ms"]
    runs = eval_config["latency_runs"]
    warmup = eval_config["latency_warmup"]
    geometry = frame_geometry(config)

    section("TASK 7 — LATENCY AND BOUNDING MEASUREMENT")
    seed_everything(config["data"].get("seed", 42))

    print(f"Budget: {budget} ms per frame (STM32 requirement) | {runs} timed runs "
          f"after {warmup} warm-up")
    print(f"Pipeline geometry: {geometry['clip_duration_s']}s window "
          f"({geometry['clip_samples']} samples @ {geometry['sample_rate']} Hz), "
          f"STFT window {geometry['window_ms']:.1f} ms, hop {geometry['hop_ms']:.1f} ms")

    devices = [resolve_device(config)]
    if devices[0].type != "cpu":
        devices.append(torch.device("cpu"))

    measurements = []
    for device in devices:
        measurement = benchmark(config, device, runs, warmup)
        stats = measurement["end_to_end"]
        measurement["budget_ms"] = budget
        measurement["headroom_vs_budget"] = budget / stats["p99_ms"]
        measurement["headroom_vs_clip_cadence"] = (
            geometry["clip_duration_s"] * 1000.0 / stats["p99_ms"])
        measurement["headroom_vs_hop"] = geometry["hop_ms"] / stats["p99_ms"]
        measurement["max_sustainable_fps"] = 1000.0 / stats["mean_ms"]
        measurement["meets_budget"] = stats["p99_ms"] < budget
        measurements.append(measurement)

        print(f"\n  {measurement['device']} ({measurement['device_name']})")
        print(f"    end-to-end : mean {stats['mean_ms']:7.3f} ms | median {stats['median_ms']:7.3f} "
              f"| min {stats['min_ms']:7.3f} | max {stats['max_ms']:7.3f} "
              f"| p95 {stats['p95_ms']:7.3f} | p99 {stats['p99_ms']:7.3f} "
              f"| sd {stats['stdev_ms']:6.3f}")
        print(f"    breakdown  : mel transform {measurement['transform_only']['mean_ms']:.3f} ms "
              f"+ CNN {measurement['model_only']['mean_ms']:.3f} ms")
        print(f"    vs {budget} ms budget    : {measurement['headroom_vs_budget']:6.1f}x headroom "
              f"on p99  [{verdict(measurement['meets_budget'])}]")
        print(f"    vs {geometry['hop_ms']:.1f} ms hop     : "
              f"{measurement['headroom_vs_hop']:6.1f}x headroom on p99")
        print(f"    vs {geometry['clip_duration_s'] * 1000:.0f} ms clip cadence: "
              f"{measurement['headroom_vs_clip_cadence']:6.1f}x headroom on p99")
        print(f"    sustainable rate      : {measurement['max_sustainable_fps']:.0f} frames/s")

    deployment = measurements[0]
    checks = {
        f"deployment_p99_under_{budget}ms": deployment["meets_budget"],
        f"deployment_mean_under_{budget}ms": deployment["end_to_end"]["mean_ms"] < budget,
        f"cpu_worst_case_under_{budget}ms": all(m["meets_budget"] for m in measurements),
        "faster_than_realtime_clip_cadence": deployment["headroom_vs_clip_cadence"] > 1.0,
    }
    status = verdict(all(checks.values()))
    print()
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 7: {status}")

    result = {
        "task": 7, "name": "Latency and bounding measurement", "status": status,
        "budget_ms": budget, "runs": runs, "warmup": warmup,
        "frame_geometry": geometry,
        "deployment_device": deployment["device"],
        "devices": [{k: v for k, v in m.items() if k != "samples_ms"} for m in measurements],
        "checks": checks,
    }
    write_csv(
        [{"device": m["device"], "stage": stage, **m[stage]}
         for m in measurements for stage in ("end_to_end", "transform_only", "model_only")],
        reports_dir(config) / "latency.csv",
    )
    result["figure"] = _plot(measurements, budget, config)
    write_json(result, reports_dir(config) / "latency.json")
    return result


def _plot(measurements, budget, config):
    """Plot the end-to-end latency distribution for each device.

    Args:
        measurements: Per-device benchmark records.
        budget: Frame budget in milliseconds, drawn as a reference line.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    new_figure()
    for measurement, color in zip(measurements, (COLORS["train"], COLORS["val"])):
        stats = measurement["end_to_end"]
        plt.hist(measurement["samples_ms"], bins=60, alpha=0.65, color=color,
                 label=f"{measurement['device']} (mean {stats['mean_ms']:.2f} ms, "
                       f"p99 {stats['p99_ms']:.2f} ms)")
    plt.axvline(budget, color=COLORS["f1"], linestyle="--", linewidth=2,
                label=f"STM32 budget ({budget} ms)")
    plt.title(f"End-to-end inference latency ({measurements[0]['end_to_end']['runs']} runs)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return finish(graphs_dir(config) / "latency_histogram.png")


if __name__ == "__main__":
    run()
