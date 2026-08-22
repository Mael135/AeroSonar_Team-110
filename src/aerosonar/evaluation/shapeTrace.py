"""Task 2: tensor dimensionality assertion.

Traces one second of real audio through every stage of the pipeline, recording the
tensor shape, dtype and value range at each step, then asserts those shapes against
what the architecture should produce.

The purpose is to establish that no data is silently truncated, padded or transposed
between stages. A mel spectrogram that loses time frames, or a flatten that folds the
wrong axis, will train without error and leave no trace in the loss curve.

Model-internal shapes are captured with forward hooks rather than print statements
inside the model, so the instrumentation cannot alter the deployed forward pass.

Run from the repository root::

    python -m aerosonar.evaluation.shapeTrace
"""
from pathlib import Path

import torch
import torchaudio

from aerosonar.config import load_default_config
from aerosonar.data.dataset import METADATA_PATH, TENSOR_DIR, SpectrogramTensorDataset
from aerosonar.evaluation.common import (load_trained_model, reports_dir, resolve_device,
                                         section, verdict, write_csv, write_json)
from aerosonar.features.transforms import SpectrogramTransform

BATCH_SIZE = 4


def _describe(name, tensor, note=""):
    """Summarise one tensor for the trace table.

    Args:
        name: Stage label.
        tensor: The tensor to describe.
        note: Optional annotation carried into the report.

    Returns:
        dict: Shape, dtype, device, element count, value range, mean, a finiteness
        flag, and the note.
    """
    flat = tensor.detach().float()
    return {
        "stage": name,
        "shape": str(tuple(tensor.shape)),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "min": round(flat.min().item(), 4),
        "max": round(flat.max().item(), 4),
        "mean": round(flat.mean().item(), 4),
        "finite": bool(torch.isfinite(flat).all().item()),
        "note": note,
    }


def _first_raw_clip(config):
    """Load one clip from the first raw recording.

    Args:
        config: Project configuration, read for the raw data path, sample rate and
            clip duration.

    Returns:
        tuple: ``(filename, waveform, sample_rate)`` where ``waveform`` has shape
        ``(1, clip_samples)``.

    Raises:
        FileNotFoundError: If the raw data directory contains no WAV files.
    """
    raw_dir = Path(config["paths"]["data_raw"])
    wav_files = sorted(raw_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files under {raw_dir}")
    waveform, sr = torchaudio.load(wav_files[0])
    n = int(config["data"]["sample_rate"] * config["data"]["duration"])
    return wav_files[0].name, waveform[:1, :n], sr


def run(config=None):
    """Trace tensor shapes through the pipeline and assert them.

    Writes ``tensor_shapes_pipeline.csv``, ``tensor_shapes_model.csv`` and
    ``shape_trace.json``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, the per-stage and per-layer shape tables,
        the total parameter count, and the individual check outcomes.
    """
    config = config or load_default_config()
    device = resolve_device(config)
    section("TASK 2 — TENSOR DIMENSIONALITY ASSERTION")

    sr = config["data"]["sample_rate"]
    n_mels = config["spectrogram"]["n_mels"]
    hop = config["spectrogram"]["hop_length"]
    clip_samples = int(sr * config["data"]["duration"])
    expected_frames = clip_samples // hop + 1  # center=True pads by n_fft//2 on both sides

    filename, waveform, file_sr = _first_raw_clip(config)
    print(f"Source clip: {filename} @ {file_sr} Hz, {config['data']['duration']}s")

    transform = SpectrogramTransform(config)
    rows = [_describe("1. raw waveform (C, N)", waveform, f"{clip_samples} samples @ {sr} Hz")]

    normalized = transform._normalize_loudness(waveform)
    rows.append(_describe("2. loudness-normalized waveform", normalized,
                          f"RMS -> {config['spectrogram']['normalize_target_dbfs']} dBFS"))

    mel = transform.mel_spectrogram(normalized)
    rows.append(_describe("3. mel spectrogram (power)", mel,
                          f"n_mels={n_mels}, hop={hop}"))

    spec = transform.amplitude_to_db(mel)
    rows.append(_describe("4. amplitude -> dB", spec, "model input, single clip"))

    stored = SpectrogramTensorDataset(METADATA_PATH, TENSOR_DIR)[0][0]
    rows.append(_describe("5. stored .pt tensor", stored, "written by preprocessData"))

    batch = spec.unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).to(device)
    rows.append(_describe("6. batched model input (B, C, F, T)", batch, f"B={BATCH_SIZE}"))

    model = load_trained_model(config, device)
    layer_rows = []
    handles = []

    def make_hook(block, index, module):
        """Build a forward hook that records one layer's input and output shapes."""
        def hook(_module, inputs, output):
            """Record this layer's shapes and parameter count."""
            params = sum(p.numel() for p in _module.parameters())
            layer_rows.append({
                "block": block,
                "index": index,
                "layer": type(module).__name__,
                "config": str(module),
                "in_shape": str(tuple(inputs[0].shape)),
                "out_shape": str(tuple(output.shape)),
                "params": params,
                "finite": bool(torch.isfinite(output).all().item()),
            })
        return hook

    for block_name in ("features", "classifier"):
        for index, module in enumerate(getattr(model, block_name)):
            handles.append(module.register_forward_hook(make_hook(block_name, index, module)))

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    for handle in handles:
        handle.remove()

    rows.append(_describe("7. logits (B, 2)", logits, "raw, pre-softmax"))
    rows.append(_describe("8. softmax probabilities (B, 2)", probs, "column 1 = drone"))
    rows.append(_describe("9. drone probability (B,)", probs[:, 1], "final scalar per clip"))

    print("\nPipeline stages:")
    for row in rows:
        print(f"  {row['stage']:42s} {row['shape']:>20s}  "
              f"[{row['min']:>10.3f}, {row['max']:>10.3f}]  {row['note']}")

    print("\nModel forward pass:")
    for row in layer_rows:
        print(f"  {row['block']:>10s}[{row['index']:2d}] {row['layer']:<18s} "
              f"{row['in_shape']:>22s} -> {row['out_shape']:<22s} {row['params']:>7d} params")

    trunk_out = layer_rows[len(model.features) - 1]["out_shape"]
    checks = {
        "mel_shape_is_1_nmels_frames": tuple(spec.shape) == (1, n_mels, expected_frames),
        "stored_tensor_matches_live_transform": tuple(stored.shape) == tuple(spec.shape),
        "conv_trunk_collapses_to_256x1x1": trunk_out == f"({BATCH_SIZE}, 256, 1, 1)",
        "output_is_two_logits": tuple(logits.shape) == (BATCH_SIZE, 2),
        "probabilities_sum_to_one": torch.allclose(
            probs.sum(dim=1), torch.ones(BATCH_SIZE, device=device), atol=1e-6),
        "all_stages_finite": all(row["finite"] for row in rows)
                             and all(row["finite"] for row in layer_rows),
    }
    status = verdict(all(checks.values()))

    print(f"\nExpected mel frames: {clip_samples} samples / hop {hop} + 1 = {expected_frames}")
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 2: {status}")

    result = {
        "task": 2, "name": "Tensor dimensionality assertion", "status": status,
        "source_clip": filename,
        "clip_samples": clip_samples,
        "expected_mel_frames": expected_frames,
        "model_input_shape": str(tuple(batch.shape)),
        "model_output_shape": str(tuple(logits.shape)),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "stages": rows,
        "layers": layer_rows,
        "checks": checks,
    }
    write_csv(rows, reports_dir(config) / "tensor_shapes_pipeline.csv")
    write_csv(layer_rows, reports_dir(config) / "tensor_shapes_model.csv")
    write_json(result, reports_dir(config) / "shape_trace.json")
    return result


if __name__ == "__main__":
    run()
