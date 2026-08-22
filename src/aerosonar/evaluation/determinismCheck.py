"""Task 3: deterministic execution check.

Runs identical audio through the deployed inference path repeatedly and requires
identical confidence scores, establishing that BatchNorm and Dropout behave correctly
at inference time. Four checks of increasing strength are applied:

A. Repeatability. Sequential ``eval()`` passes on identical input must agree
   bit-for-bit, which detects an active Dropout or a stray RNG call in the deployed
   path.
B. Batch invariance. A clip scored alone must match the same clip scored as the first
   element of a batch padded with deliberately extreme companions. This is the
   substantive BatchNorm test: a model left in ``train()`` mode normalises using batch
   statistics, so its answer for one clip would depend on whatever else arrived with it.
C. Train-mode control. Repeated ``train()`` passes must differ. Without this, checks A
   and B would also pass on a model containing no stochastic layers, and would
   establish nothing about ``eval()`` having any effect.
D. BatchNorm margin. The residual from check B must be orders of magnitude smaller than
   the batch-statistics effect measured in check C.

Check B is exactly zero on CPU but not on CUDA, where cuDNN selects different
convolution algorithms for different batch shapes. The resulting change in summation
order perturbs the final significant digits. This is a property of the GPU kernel
library rather than of the model, which is why check B applies a tolerance and check D
exists to confirm the residual is numerical noise rather than a state leak. Both devices
are measured so the two can be compared directly.

Run from the repository root::

    python -m aerosonar.evaluation.determinismCheck
"""
from pathlib import Path

import torch
import torchaudio

from aerosonar.config import load_default_config
from aerosonar.evaluation.common import (load_deployed_threshold, load_trained_model,
                                         reports_dir, resolve_device, section, verdict,
                                         write_json)
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.utils.seeding import seed_everything

N_REPEATS = 3
BATCH_CONTEXT = 8

#: Bound on batch-shape-dependent floating-point noise from GPU kernel selection. A
#: genuine batch-statistics leak moves the score by order 0.1, some four orders of
#: magnitude above this.
BATCH_TOLERANCE = 1e-4

#: Factor by which the batch-statistics effect must exceed the observed batch-context
#: residual for that residual to count as numerical noise.
BATCHNORM_MARGIN = 100.0


def _load_clip(config, device):
    """Load a fixed clip from a real recording.

    Uses recorded audio rather than synthesised input so the measurement covers the same
    bytes on every run.

    Args:
        config: Project configuration.
        device: Device to place the waveform on.

    Returns:
        tuple: ``(filename, waveform)`` with waveform shape ``(1, clip_samples)``.

    Raises:
        FileNotFoundError: If the raw data directory contains no WAV files.
    """
    raw_dir = Path(config["paths"]["data_raw"])
    wav_files = sorted(raw_dir.rglob("*drone*.wav")) or sorted(raw_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files under {raw_dir}")
    waveform, _ = torchaudio.load(wav_files[0])
    n = int(config["data"]["sample_rate"] * config["data"]["duration"])
    # Offset into the file: the opening second of a recording is often near-silent.
    start = min(n * 30, max(0, waveform.shape[-1] - n))
    return wav_files[0].name, waveform[:1, start:start + n].to(device)


def measure_device(config, device, threshold):
    """Run all four determinism checks on one device.

    Args:
        config: Project configuration.
        device: Device to measure.
        threshold: Detection threshold, used to record the resulting decision.

    Returns:
        dict: Per-run probabilities, the batch-context and batch-statistics
        measurements, the train-mode control values, the individual check outcomes, and
        an overall ``status``.
    """
    filename, clip = _load_clip(config, device)
    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(device)
    transform.amplitude_to_db = transform.amplitude_to_db.to(device)
    model = load_trained_model(config, device)

    print(f"\n--- device: {device} --- ({filename}, model.training = {model.training})")

    runs = []
    for i in range(N_REPEATS):
        with torch.no_grad():
            spec = transform(clip).unsqueeze(0).to(device)
            logits = model(spec)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        runs.append({
            "run": i + 1, "drone_probability": prob, "repr": repr(prob),
            "hex": float(prob).hex(), "logits": [v.item() for v in logits[0]],
            "detected": bool(prob > threshold),
        })
        print(f"  run {i + 1}: p(drone) = {prob:.17f}  ({float(prob).hex()})")

    with torch.no_grad():
        spec = transform(clip).unsqueeze(0).to(device)
        alone = torch.softmax(model(spec), dim=1)[0, 1].item()
        # Deliberately extreme companions: if BatchNorm were pooling batch statistics,
        # these would drag the shared mean/variance far off and move the first clip's score.
        companions = [torch.randn_like(spec) * 10 for _ in range(BATCH_CONTEXT - 1)]
        padded = torch.cat([spec] + companions)
        in_batch = torch.softmax(model(padded), dim=1)[0, 1].item()

        # In train() mode BatchNorm does use batch statistics, which calibrates how
        # large a genuine state leak would appear on this exact input.
        model.train()
        batch_stat = torch.softmax(model(padded), dim=1)[0, 1].item()
        train_runs = [torch.softmax(model(spec), dim=1)[0, 1].item() for _ in range(N_REPEATS)]
        model.eval()

    batch_delta = abs(alone - in_batch)
    batch_stat_delta = abs(alone - batch_stat)
    print(f"  alone in a batch of 1  : {alone:.17f}")
    print(f"  first of a batch of {BATCH_CONTEXT}  : {in_batch:.17f}")
    ratio = (batch_stat_delta / batch_delta) if batch_delta else None
    margin = f" ({ratio:.0f}x larger)" if ratio else ""
    print(f"  batch-context delta    : {batch_delta:.3e} "
          f"({'bit-exact' if batch_delta == 0 else 'float noise'})")
    print(f"  batch-statistics delta : {batch_stat_delta:.3e}{margin} "
          f"— what a real BatchNorm state leak would cost on this input")
    print(f"  train-mode control     : {', '.join(f'{p:.6f}' for p in train_runs)}")

    checks = {
        f"{N_REPEATS}_eval_runs_bit_identical": len({r["hex"] for r in runs}) == 1,
        "detection_decision_stable": len({r["detected"] for r in runs}) == 1,
        "model_in_eval_mode": not model.training,
        f"batch_context_delta_under_{BATCH_TOLERANCE}": batch_delta < BATCH_TOLERANCE,
        "batchnorm_uses_running_stats": batch_stat_delta > BATCHNORM_MARGIN * batch_delta,
        "train_mode_control_varies": len({float(p).hex() for p in train_runs}) > 1,
    }
    for name, ok in checks.items():
        print(f"    [{verdict(ok)}] {name}")

    return {
        "device": str(device),
        "clip": filename,
        "probability": runs[0]["drone_probability"],
        "probability_repr": runs[0]["repr"],
        "runs": runs,
        "batch_context": {
            "batch_of_1": alone,
            f"first_of_batch_of_{BATCH_CONTEXT}": in_batch,
            "delta": batch_delta,
            "bit_identical": batch_delta == 0.0,
        },
        "batch_statistics_control": {
            "probability": batch_stat,
            "delta_vs_alone": batch_stat_delta,
            "ratio_vs_batch_context_delta": (batch_stat_delta / batch_delta) if batch_delta else None,
        },
        "train_mode_control": train_runs,
        "checks": checks,
        "status": verdict(all(checks.values())),
    }


def run(config=None):
    """Run the determinism check on every available device.

    The deployment device is measured first. Where that is CUDA, the CPU is measured as
    well to provide a bit-exact reference.

    Writes ``determinism.json``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, the reported probability, per-device
        measurements, and the deployment device's check outcomes.
    """
    config = config or load_default_config()
    section("TASK 3 — DETERMINISTIC EXECUTION CHECK")
    seed_everything(config["data"].get("seed", 42), deterministic=True)
    threshold = load_deployed_threshold(config)

    devices = [resolve_device(config)]
    if devices[0].type != "cpu":
        devices.append(torch.device("cpu"))

    measurements = [measure_device(config, device, threshold) for device in devices]
    deployment = measurements[0]

    status = verdict(all(m["status"] == "PASS" for m in measurements))
    print(f"\nTask 3: {status}")

    result = {
        "task": 3, "name": "Deterministic execution check", "status": status,
        "threshold": threshold,
        "n_repeats": N_REPEATS,
        "batch_tolerance": BATCH_TOLERANCE,
        "deployment_device": deployment["device"],
        "probability": deployment["probability"],
        "probability_repr": deployment["probability_repr"],
        "devices": measurements,
        "checks": deployment["checks"],
    }
    write_json(result, reports_dir(config) / "determinism.json")
    return result


if __name__ == "__main__":
    run()
