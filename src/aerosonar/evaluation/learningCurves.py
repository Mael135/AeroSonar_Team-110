"""Task 8: generalization and domain-shift analysis.

Plots training loss against validation loss over the epochs and turns the shape of those
curves into a named diagnosis. The distinction it draws is between two failures that a
single accuracy figure cannot separate:

Overfitting
    The model memorises the training clips. Training loss continues to fall while
    validation loss turns upward. This is visible in the curves alone.

Domain shift
    The model generalises acceptably to held-out clips that resemble the training data
    and still fails in deployment, because the curated corpus never contained the
    conditions deployment presents. This is *not* visible in the curves. It appears as a
    model that scores well in aggregate yet loses discriminative power once a
    confounding cue is held constant.

The second is diagnosed by importing Task 5's per-location result, which is the only
place the confound can be measured.

Run from the repository root::

    python -m aerosonar.evaluation.learningCurves
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aerosonar.config import load_default_config
from aerosonar.evaluation.common import (SKIP, graphs_dir, reports_dir, section, verdict,
                                         write_json)
from aerosonar.utils.plotting import COLORS, finish, new_figure

#: Ratio of final to minimum validation loss above which the upturn counts as genuine
#: rather than epoch-to-epoch noise.
OVERFIT_RATIO = 1.05

#: Final training accuracy below which the model is judged not to have fit its data.
UNDERFIT_TRAIN_ACC = 80.0

#: Matthews correlation on the un-confounded subset at or below which the model has no
#: usable discriminative power there, whatever its aggregate accuracy shows.
CONFOUND_MCC = 0.1


def load_history(path):
    """Read the per-epoch training history written by the training run.

    Args:
        path: Path to ``train_history.csv``.

    Returns:
        list[dict] | None: One record per epoch with numeric fields, or None if the
        file does not exist.
    """
    if not Path(path).exists():
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in row.items():
            row[key] = int(value) if key == "epoch" else float(value)
    return rows


def diagnose(history, confound):
    """Derive a named diagnosis from the loss curves and the confound evidence.

    Overfitting is identified when validation loss rises materially above its minimum
    while training loss is still falling. Underfitting is identified from the final
    training accuracy. Domain shift is identified from the un-confounded subset's
    Matthews correlation, since the loss curves cannot reveal it.

    Args:
        history: Per-epoch records from :func:`load_history`.
        confound: Task 5's metrics pooled over locations containing both classes, or
            None when that result is unavailable.

    Returns:
        dict: The verdict together with the quantities supporting it, including the
        epoch of minimum validation loss, the generalization gap and the final metrics.
    """
    train_loss = np.array([h["train_loss"] for h in history])
    val_loss = np.array([h["val_loss"] for h in history])
    train_acc = np.array([h["train_acc"] for h in history])

    best_epoch = int(np.argmin(val_loss)) + 1
    final_gap = float(val_loss[-1] - train_loss[-1])
    val_upturn = float(val_loss[-1] / val_loss.min())
    # Compare halves rather than adjacent epochs, since single-epoch deltas are noise.
    half = max(1, len(train_loss) // 2)
    train_still_falling = bool(train_loss[half:].mean() < train_loss[:half].mean())

    findings = []
    if val_upturn > OVERFIT_RATIO and train_still_falling:
        findings.append("overfitting")
    if train_acc[-1] < UNDERFIT_TRAIN_ACC:
        findings.append("underfitting")
    if confound is not None and confound["mcc"] <= CONFOUND_MCC:
        findings.append("domain-shift / confounded-feature")

    return {
        "verdict": " + ".join(findings) if findings else "well-fit",
        "epochs": len(history),
        "best_val_loss_epoch": best_epoch,
        "epochs_trained_past_optimum": len(history) - best_epoch,
        "min_val_loss": float(val_loss.min()),
        "final_val_loss": float(val_loss[-1]),
        "final_train_loss": float(train_loss[-1]),
        "final_generalization_gap": final_gap,
        "val_loss_upturn_ratio": val_upturn,
        "train_loss_still_falling": train_still_falling,
        "final_train_acc": float(train_acc[-1]),
        "final_val_acc": float(history[-1]["val_acc"]),
        "final_val_f1": float(history[-1]["val_f1"]),
        "best_val_f1": float(max(h["val_f1"] for h in history)),
    }


def run(config=None):
    """Analyse the training curves and diagnose the failure mode.

    Skips with a message if no training history exists, since the history is produced by
    the training run rather than by this module.

    Writes ``generalization.json``, ``loss_curves.png`` and ``accuracy_curves.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, the diagnosis and its supporting
        quantities, the un-confounded subset metrics, and the individual check
        outcomes.
    """
    config = config or load_default_config()
    section("TASK 8 — GENERALIZATION vs DOMAIN-SHIFT ANALYSIS")

    history_path = reports_dir(config) / "train_history.csv"
    history = load_history(history_path)
    if not history:
        print(f"  No training history at {history_path}. "
              f"Run `python -m aerosonar.training.trainCNN` first.")
        return {"task": 8, "name": "Generalization vs domain shift", "status": SKIP,
                "reason": f"missing {history_path}"}

    # Task 5's un-confounded subset, where that check has already run.
    confound = None
    metrics_path = reports_dir(config) / "test_metrics.json"
    if metrics_path.exists():
        confound = json.loads(metrics_path.read_text()).get("pooled_both_class_locations")

    analysis = diagnose(history, confound)

    print(f"Epochs: {analysis['epochs']} | best validation loss at epoch "
          f"{analysis['best_val_loss_epoch']} "
          f"({analysis['epochs_trained_past_optimum']} epochs trained past it)")
    print(f"  final train loss   : {analysis['final_train_loss']:.4f}")
    print(f"  final val loss     : {analysis['final_val_loss']:.4f} "
          f"(min {analysis['min_val_loss']:.4f}, "
          f"{(analysis['val_loss_upturn_ratio'] - 1) * 100:+.1f}% off its minimum)")
    print(f"  generalization gap : {analysis['final_generalization_gap']:+.4f}")
    print(f"  final train acc    : {analysis['final_train_acc']:.2f}%")
    print(f"  final val acc      : {analysis['final_val_acc']:.2f}%  "
          f"(F1 {analysis['final_val_f1']:.4f}, best {analysis['best_val_f1']:.4f})")

    if confound is not None:
        print(f"\n  Un-confounded subset (test locations containing both classes):")
        print(f"    MCC {confound['mcc']:.4f} | precision {confound['precision']:.4f} | "
              f"recall {confound['recall']:.4f} | specificity {confound['specificity']:.4f}")
        if confound["mcc"] <= CONFOUND_MCC:
            print(f"    -> At MCC {confound['mcc']:.3f} the model has no discriminative power "
                  f"once location is held constant: it separates recording environments, "
                  f"not drone from no-drone.")
    else:
        print("\n  No Task 5 result available — confound evidence not incorporated.")

    print(f"\n  DIAGNOSIS: {analysis['verdict']}")

    checks = {
        "history_available": True,
        "validation_loss_did_not_diverge":
            analysis["val_loss_upturn_ratio"] <= OVERFIT_RATIO,
        "model_fits_training_data": analysis["final_train_acc"] >= UNDERFIT_TRAIN_ACC,
        "generalizes_within_location":
            confound is not None and confound["mcc"] > CONFOUND_MCC,
    }
    status = verdict(all(checks.values()))
    print()
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 8: {status}")

    result = {
        "task": 8, "name": "Generalization vs domain shift", "status": status,
        "analysis": analysis,
        "unconfounded_subset": confound,
        "thresholds": {
            "overfit_val_loss_ratio": OVERFIT_RATIO,
            "underfit_train_acc": UNDERFIT_TRAIN_ACC,
            "confound_mcc": CONFOUND_MCC,
        },
        "checks": checks,
    }
    result["loss_figure"] = _plot_loss(history, analysis, config)
    result["accuracy_figure"] = _plot_accuracy(history, config)
    write_json(result, reports_dir(config) / "generalization.json")
    return result


def _plot_loss(history, analysis, config):
    """Plot training against validation loss, shading the generalization gap.

    Args:
        history: Per-epoch records.
        analysis: Diagnosis from :func:`diagnose`, read for the title and the marked
            epoch of minimum validation loss.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    epochs = [h["epoch"] for h in history]
    new_figure()
    plt.plot(epochs, [h["train_loss"] for h in history], color=COLORS["train"],
             linewidth=2, label="Train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], color=COLORS["val"],
             linewidth=2, label="Validation loss")
    plt.axvline(analysis["best_val_loss_epoch"], color=COLORS["reference"], linestyle="--",
                label=f"Min val loss (epoch {analysis['best_val_loss_epoch']})")
    plt.fill_between(epochs, [h["train_loss"] for h in history],
                     [h["val_loss"] for h in history], color=COLORS["f1"], alpha=0.12,
                     label="Generalization gap")
    # Verdict on its own line: it can be long enough to run off the canvas inline.
    plt.title(f"Training vs Validation Loss\ndiagnosis: {analysis['verdict']}", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return finish(graphs_dir(config) / "loss_curves.png")


def _plot_accuracy(history, config):
    """Plot training and validation accuracy, with validation F1 for reference.

    Args:
        history: Per-epoch records.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    epochs = [h["epoch"] for h in history]
    new_figure()
    plt.plot(epochs, [h["train_acc"] for h in history], color=COLORS["train"],
             linewidth=2, label="Train accuracy")
    plt.plot(epochs, [h["val_acc"] for h in history], color=COLORS["val"],
             linewidth=2, label="Validation accuracy")
    plt.plot(epochs, [h["val_f1"] * 100 for h in history], color=COLORS["f1"],
             linewidth=2, linestyle="--", label="Validation F1 (x100)")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Percent")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    return finish(graphs_dir(config) / "accuracy_curves.png")


if __name__ == "__main__":
    run()
