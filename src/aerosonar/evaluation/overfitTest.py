"""Task 1: single-batch overfit test.

Trains a freshly initialised model on one small batch until it memorises it. A network
whose loss function, optimiser and gradient path are wired correctly can always fit a
handful of examples; failure to do so indicates a structural defect such as a detached
computation graph, a frozen parameter, or a label-to-logit mismatch.

This is a check of implementation correctness. It establishes nothing about detection
accuracy on real data.

Run from the repository root::

    python -m aerosonar.evaluation.overfitTest
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from aerosonar.config import load_default_config
from aerosonar.data.dataset import (METADATA_PATH, TENSOR_DIR, SpectrogramTensorDataset,
                                    load_joined_metadata, split_file_ids)
from aerosonar.evaluation.common import (graphs_dir, reports_dir, resolve_device, section,
                                         verdict, write_csv, write_json)
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.utils.plotting import COLORS, finish, new_figure
from aerosonar.utils.seeding import seed_everything

#: Loss below which the batch counts as memorised. Exact zero is unreachable in
#: float32 arithmetic, so the check needs an explicit tolerance.
LOSS_TOLERANCE = 1e-3


def select_batch(batch_size, metadata_path=METADATA_PATH, tensor_dir=TENSOR_DIR):
    """Assemble a small class-balanced batch from the training split.

    Augmentation is bypassed. Cross-session mixing and SpecAugment present a different
    view of each clip on every access, so an augmented batch has no fixed target to
    memorise and a correctly implemented model would appear to fail.

    Args:
        batch_size: Total number of clips, divided evenly between the two classes.
        metadata_path: Path to ``metadata.csv``.
        tensor_dir: Directory holding the ``.pt`` tensors.

    Returns:
        tuple: ``(samples, labels, n_recordings)`` where ``samples`` has shape
        ``(batch_size, 1, n_mels, frames)``, ``labels`` is a ``torch.long`` tensor, and
        ``n_recordings`` counts the distinct source recordings represented.
    """
    meta = load_joined_metadata(metadata_path)
    train_ids, _, _ = split_file_ids(meta)
    train_meta = meta[meta["file_id"].isin(train_ids)]

    per_class = batch_size // 2
    picks = []
    for target in (1, 0):
        rows = train_meta[train_meta["target"] == target]
        # Stride through the rows so the batch spans recordings. Consecutive chunks of
        # one file are near-duplicates and would make memorisation trivially easy.
        picks.append(rows.iloc[:: max(1, len(rows) // per_class)].head(per_class))

    base = SpectrogramTensorDataset(metadata_file=metadata_path, data_dir=tensor_dir)
    indices = [i for part in picks for i in part.index.tolist()]
    samples = torch.stack([base[i][0] for i in indices])
    labels = torch.stack([base[i][1] for i in indices])
    n_recordings = len({fid for part in picks for fid in part["file_id"]})
    return samples, labels, n_recordings


def run(config=None):
    """Run the overfit test and write its artifacts.

    Trains a fresh model on a single fixed batch for the configured number of epochs,
    at a learning rate high enough to converge in that budget. The production rate is
    far lower and could not memorise a batch in this time, which is immaterial here
    because the subject of the test is gradient flow rather than the training schedule.

    Memorisation is assessed in ``eval()`` mode. The classifier's ``Dropout(0.5)``
    keeps the training-mode loss stochastic and bounded away from zero even for a fully
    memorised batch.

    Writes ``overfit_curve.csv``, ``overfit_test.json`` and ``overfit_curve.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, the initial and final loss, the epoch at
        which accuracy first reached 100 percent, and the individual check outcomes.
    """
    config = config or load_default_config()
    eval_config = config["evaluation"]
    epochs = eval_config["overfit_epochs"]
    batch_size = eval_config["overfit_batch_size"]
    lr = eval_config["overfit_lr"]
    device = resolve_device(config)

    section("TASK 1 — SINGLE-BATCH OVERFIT TEST (gradient sanity check)")
    seed_everything(config["data"].get("seed", 42), deterministic=True)

    samples, labels, n_recordings = select_batch(batch_size)
    samples = samples.to(device).float()
    labels = labels.to(device).long()
    print(f"Batch: {tuple(samples.shape)} from {n_recordings} distinct recordings | "
          f"labels {labels.tolist()}")
    print(f"Training {epochs} epochs at lr={lr} (production lr is "
          f"{config['training']['lr']}, far too small to memorise in {epochs} epochs — "
          f"this check is about gradient flow, not the production schedule).")

    model = SpectrogramCNN().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(config["training"]["class_weights"]).to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr)

    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(samples), labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(samples)
            eval_loss = criterion(logits, labels).item()
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item() * 100

        history.append({
            "epoch": epoch + 1, "train_loss": loss.item(), "eval_loss": eval_loss,
            "accuracy": accuracy, "grad_norm": grad_norm,
        })
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1:4d} | train_loss={loss.item():.6f} "
                  f"eval_loss={eval_loss:.6f} acc={accuracy:6.2f}% grad_norm={grad_norm:.3e}")

    first, last = history[0], history[-1]
    checks = {
        "accuracy_100pct": last["accuracy"] == 100.0,
        f"eval_loss_below_{LOSS_TOLERANCE}": last["eval_loss"] < LOSS_TOLERANCE,
        "loss_reduced_by_99pct": last["eval_loss"] < 0.01 * first["eval_loss"],
        "gradients_nonzero": all(h["grad_norm"] > 0 for h in history[:10]),
        "no_nan_loss": all(h["eval_loss"] == h["eval_loss"] for h in history),
    }
    status = verdict(all(checks.values()))

    print(f"\nInitial eval loss : {first['eval_loss']:.6f}")
    print(f"Final eval loss   : {last['eval_loss']:.3e}")
    print(f"Final accuracy    : {last['accuracy']:.2f}%")
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 1: {status}")

    result = {
        "task": 1, "name": "Single-batch overfit test", "status": status,
        "batch_size": batch_size, "epochs": epochs, "lr": lr,
        "n_recordings_in_batch": n_recordings,
        "loss_tolerance": LOSS_TOLERANCE,
        "initial_eval_loss": first["eval_loss"],
        "final_eval_loss": last["eval_loss"],
        "final_train_loss": last["train_loss"],
        "final_accuracy_pct": last["accuracy"],
        "epochs_to_100pct": next(
            (h["epoch"] for h in history if h["accuracy"] == 100.0), None
        ),
        "checks": checks,
    }

    write_csv(history, reports_dir(config) / "overfit_curve.csv")
    result["figure"] = _plot(history, config)
    write_json(result, reports_dir(config) / "overfit_test.json")
    return result


def _plot(history, config):
    """Plot loss and accuracy against epoch.

    Both the training-mode and evaluation-mode losses are drawn, on a log scale, so the
    gap attributable to dropout is visible.

    Args:
        history: Per-epoch records from :func:`run`.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    epochs = [h["epoch"] for h in history]
    fig = new_figure()
    ax = fig.gca()
    ax.plot(epochs, [h["train_loss"] for h in history], color=COLORS["train"],
            linewidth=1, alpha=0.5, label="Loss (train mode, dropout active)")
    ax.plot(epochs, [h["eval_loss"] for h in history], color=COLORS["f1"],
            linewidth=2, label="Loss (eval mode)")
    ax.axhline(LOSS_TOLERANCE, color=COLORS["reference"], linestyle=":",
               label=f"Zero tolerance ({LOSS_TOLERANCE})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss (log scale)")
    ax.grid(True, linestyle="--", alpha=0.6)

    twin = ax.twinx()
    twin.plot(epochs, [h["accuracy"] for h in history], color=COLORS["accuracy"],
              linewidth=2, label="Accuracy")
    twin.set_ylabel("Accuracy (%)")
    twin.set_ylim(-5, 105)
    twin.grid(False)

    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="center right")
    plt.title(f"Single-batch overfit ({len(history)} epochs)")
    return finish(graphs_dir(config) / "overfit_curve.png")


if __name__ == "__main__":
    run()
