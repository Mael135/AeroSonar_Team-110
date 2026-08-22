"""Shared helpers for the verification and evaluation suite.

Every check in this package exposes ``run(config) -> dict``, printing a human-readable
summary and returning a structured result that includes a ``status`` field. Each check
also writes its own artifacts, so it can be run alone or under
:mod:`aerosonar.evaluation.runAll`.
"""
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio
import yaml

from aerosonar.models.spectrogramCNN import SpectrogramCNN

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def resolve_device(config=None, prefer_cuda=True):
    """Select the compute device.

    Args:
        config: Unused; accepted so callers can pass the configuration uniformly.
        prefer_cuda: Whether to use CUDA when it is available.

    Returns:
        torch.device: CUDA if requested and available, otherwise CPU.
    """
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def reports_dir(config):
    """Return the report output directory, creating it if needed.

    Args:
        config: Project configuration, read for ``evaluation.reports_dir``.

    Returns:
        pathlib.Path: The directory.
    """
    path = Path(config["evaluation"]["reports_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def graphs_dir(config):
    """Return the figure output directory, creating it if needed.

    Args:
        config: Project configuration, read for ``evaluation.graphs_dir``.

    Returns:
        pathlib.Path: The directory.
    """
    path = Path(config["evaluation"]["graphs_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def weights_path(config):
    """Return the path to the trained model checkpoint.

    Args:
        config: Project configuration, read for ``paths.weights``.

    Returns:
        pathlib.Path: Path to ``CNN_best.pth``.
    """
    return Path(config["paths"]["weights"]) / "CNN_best.pth"


def load_trained_model(config, device=None):
    """Load the trained checkpoint into an evaluation-mode model.

    Args:
        config: Project configuration.
        device: Target device. Defaults to :func:`resolve_device`.

    Returns:
        SpectrogramCNN: The loaded model, in ``eval()`` mode.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    device = device or resolve_device(config)
    path = weights_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained weights at {path}. Run `python -m aerosonar.training.trainCNN` first."
        )
    model = SpectrogramCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_deployed_threshold(config, default=0.5):
    """Read the detection threshold the deployed pipeline uses.

    Mirrors :func:`aerosonar.inference.inference.load_threshold` without importing that
    module, which depends on the optional ``sounddevice`` package.

    Args:
        config: Project configuration, read for ``paths.weights``.
        default: Value returned when ``threshold.yaml`` is absent.

    Returns:
        float: The detection threshold.
    """
    path = Path(config["paths"]["weights"]) / "threshold.yaml"
    if path.exists():
        return float(yaml.safe_load(path.read_text()).get("detection_threshold", default))
    return default


def section(title):
    """Print a banner heading.

    Args:
        title: Heading text.
    """
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def verdict(condition):
    """Convert a boolean into a status string.

    Args:
        condition: The check result.

    Returns:
        str: ``PASS`` or ``FAIL``.
    """
    return PASS if condition else FAIL


def write_json(payload, path):
    """Write a payload to a JSON file.

    Args:
        payload: JSON-serialisable object. Values that are not natively serialisable
            are coerced with ``str``.
        path: Destination file. Parent directories are created as needed.

    Returns:
        str: The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Wrote {path}")
    return str(path)


def write_csv(rows, path, fieldnames=None):
    """Write a list of dictionaries to a CSV file.

    Args:
        rows: Records to write. An empty list writes nothing.
        path: Destination file. Parent directories are created as needed.
        fieldnames: Column order. Defaults to the keys of the first row.

    Returns:
        str: The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  No rows to write for {path}")
        return str(path)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path} ({len(rows)} rows)")
    return str(path)


def _git(*args):
    """Run a git command, returning its output or ``"unknown"`` on any failure.

    Args:
        *args: Arguments passed to ``git``.

    Returns:
        str: Trimmed standard output.
    """
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"


def provenance(config):
    """Capture the code state and environment the measurements were taken under.

    A reported metric is only reproducible if the commit, device and library versions
    that produced it are recorded alongside it.

    Args:
        config: Project configuration.

    Returns:
        dict: Commit, branch, working-tree cleanliness, interpreter and library
        versions, platform, device details, and the deployed detection threshold.
    """
    device = resolve_device(config)
    return {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "detection_threshold": load_deployed_threshold(config),
    }


def frame_geometry(config):
    """Derive the timing of the analysis window from the configuration.

    Args:
        config: Project configuration, read for ``data`` and ``spectrogram``.

    Returns:
        dict: Sample rate, clip duration in seconds and samples, STFT window and hop
        lengths in milliseconds, and the mel band count.
    """
    sr = config["data"]["sample_rate"]
    spec = config["spectrogram"]
    return {
        "sample_rate": sr,
        "clip_duration_s": config["data"]["duration"],
        "clip_samples": int(sr * config["data"]["duration"]),
        "window_ms": 1000.0 * spec["win_length"] / sr,
        "hop_ms": 1000.0 * spec["hop_length"] / sr,
        "n_mels": spec["n_mels"],
    }
