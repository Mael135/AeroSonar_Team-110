"""Task 6: threshold sweep with ROC and precision-recall curves.

A classifier does not have a single accuracy but a curve of operating points. This
module evaluates every cutoff from 0 to 1 and records the full confusion matrix at each,
which is what substantiates the deployed threshold rather than merely asserting it.

Both held-out splits are swept, and the distinction matters. Validation is where the
threshold was selected, so its curve is the evidence for that choice. Test played no part
in the selection, so its curve estimates what the chosen threshold delivers on unseen
data. Reporting only the first would be circular; reporting only the second would omit
the selection criterion.

Areas under both curves are integrated over the same points that are plotted, so a
reported area always corresponds to the curve shown beside it.

Run from the repository root::

    python -m aerosonar.evaluation.thresholdSweep
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from aerosonar.config import load_default_config
from aerosonar.data.dataset import load_joined_metadata, split_file_ids
from aerosonar.evaluation.common import (graphs_dir, load_deployed_threshold,
                                         load_trained_model, reports_dir, resolve_device,
                                         section, verdict, write_csv, write_json)
from aerosonar.evaluation.metrics import score_split
from aerosonar.training.trainCNN import find_best_threshold
from aerosonar.utils.plotting import COLORS, finish, new_figure

#: Step of the threshold grid reported as a table.
COARSE_STEP = 0.05

#: Step of the denser grid used to draw smooth curves.
FINE_STEP = 0.005


def sweep(probs, labels, thresholds):
    """Evaluate the confusion matrix and derived rates across a set of thresholds.

    Args:
        probs: Drone probability per chunk.
        labels: Ground-truth labels, 1 for drone.
        thresholds: Cutoffs to evaluate.

    Returns:
        list[dict]: One record per threshold, holding the four confusion counts, true
        and false positive rates, precision, F1 and accuracy.
    """
    rows = []
    for threshold in thresholds:
        predicted = probs > threshold
        TP = int(np.sum(predicted & (labels == 1)))
        FP = int(np.sum(predicted & (labels == 0)))
        TN = int(np.sum(~predicted & (labels == 0)))
        FN = int(np.sum(~predicted & (labels == 1)))
        tpr = TP / (TP + FN) if (TP + FN) else 0.0
        fpr = FP / (FP + TN) if (FP + TN) else 0.0
        # Where no chunk exceeds the threshold, precision is defined as 1.0: no
        # predictions were made, so none were incorrect. Using 0 would impose a false
        # floor on the precision-recall curve.
        precision = TP / (TP + FP) if (TP + FP) else 1.0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0
        rows.append({
            "threshold": round(float(threshold), 4),
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "TPR_recall": round(tpr, 6), "FPR": round(fpr, 6),
            "precision": round(precision, 6), "f1": round(f1, 6),
            "accuracy": round((TP + TN) / len(labels), 6),
        })
    return rows


def roc_auc(rows):
    """Integrate the area under the ROC curve.

    Args:
        rows: Sweep records from :func:`sweep`.

    Returns:
        float: Area under the curve, with both corners anchored so a sweep that never
        reaches a false positive rate of 0 or 1 is not integrated over a truncated
        interval.
    """
    fpr = np.array([r["FPR"] for r in rows])
    tpr = np.array([r["TPR_recall"] for r in rows])
    order = np.argsort(fpr)
    fpr = np.concatenate([[0.0], fpr[order], [1.0]])
    tpr = np.concatenate([[0.0], tpr[order], [1.0]])
    return float(np.trapezoid(tpr, fpr))


def average_precision(rows):
    """Integrate the area under the precision-recall curve.

    Args:
        rows: Sweep records from :func:`sweep`.

    Returns:
        float: Average precision.
    """
    recall = np.array([r["TPR_recall"] for r in rows])
    precision = np.array([r["precision"] for r in rows])
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def analyse(name, rows_frame, threshold):
    """Sweep one split and print its summary and threshold grid.

    Args:
        name: Split name, used for labelling.
        rows_frame: Scored rows with ``prob`` and ``target`` columns.
        threshold: The deployed threshold, evaluated as a distinct operating point.

    Returns:
        tuple: ``(summary, coarse, fine)`` where ``summary`` holds the areas under both
        curves and the operating point at the deployed threshold, and the other two are
        sweep records at the coarse and fine grids.
    """
    probs = rows_frame["prob"].to_numpy()
    labels = rows_frame["target"].to_numpy().astype(int)

    coarse = sweep(probs, labels, np.round(np.arange(0.0, 1.0 + COARSE_STEP, COARSE_STEP), 4))
    fine = sweep(probs, labels, np.round(np.arange(0.0, 1.0 + FINE_STEP, FINE_STEP), 4))
    at_deployed = sweep(probs, labels, [threshold])[0]
    best_threshold, best_metrics, _ = find_best_threshold(
        torch.tensor(probs), torch.tensor(labels)
    )

    summary = {
        "split": name,
        "chunks": len(labels),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "roc_auc": roc_auc(fine),
        "average_precision": average_precision(fine),
        "positive_rate": float(labels.mean()),
        "deployed_threshold": threshold,
        "at_deployed_threshold": at_deployed,
        "best_f1_threshold_on_this_split": best_threshold,
        "best_f1_on_this_split": best_metrics["f1"],
    }

    print(f"\n  {name.upper()} ({len(labels)} chunks, {int(labels.sum())} drone)")
    print(f"    ROC-AUC            : {summary['roc_auc']:.4f}   "
          f"(0.5 = coin flip, 1.0 = perfect separation)")
    print(f"    Average precision  : {summary['average_precision']:.4f}   "
          f"(no-skill baseline = {labels.mean():.4f}, the positive rate)")
    print(f"    Best-F1 threshold  : {best_threshold:.2f} -> F1 {best_metrics['f1']:.4f}")
    print(f"    @ deployed {threshold:.2f}     : F1 {at_deployed['f1']:.4f}  "
          f"precision {at_deployed['precision']:.4f}  recall {at_deployed['TPR_recall']:.4f}  "
          f"FPR {at_deployed['FPR']:.4f}")

    print(f"\n    Threshold grid (step {COARSE_STEP}):")
    print(f"    {'thr':>5} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'TPR':>7} {'FPR':>7} "
          f"{'prec':>7} {'F1':>7}")
    for row in coarse:
        print(f"    {row['threshold']:5.2f} {row['TP']:5d} {row['FP']:5d} {row['TN']:5d} "
              f"{row['FN']:5d} {row['TPR_recall']:7.4f} {row['FPR']:7.4f} "
              f"{row['precision']:7.4f} {row['f1']:7.4f}")

    return summary, coarse, fine


def run(config=None):
    """Sweep thresholds on both held-out splits and plot the curves.

    Writes ``threshold_sweep.csv``, ``threshold_sweep.json``, ``roc_curve.png`` and
    ``pr_curve.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, per-split summaries and the individual
        check outcomes.
    """
    config = config or load_default_config()
    device = resolve_device(config)
    section("TASK 6 — THRESHOLD SWEEP (ROC / PR curves)")

    model = load_trained_model(config, device)
    threshold = load_deployed_threshold(config)
    meta = load_joined_metadata()
    _, val_ids, test_ids = split_file_ids(meta)

    print(f"Deployed threshold: {threshold:.2f}")
    results, coarse_rows, curves = {}, [], {}
    for name, file_ids in (("val", val_ids), ("test", test_ids)):
        frame = score_split(config, model, device, file_ids, meta)
        summary, coarse, fine = analyse(name, frame, threshold)
        results[name] = summary
        curves[name] = fine
        coarse_rows.extend({"split": name, **row} for row in coarse)

    test = results["test"]
    checks = {
        "test_roc_auc_above_chance": test["roc_auc"] > 0.5,
        "test_ap_above_no_skill_baseline": test["average_precision"] > test["positive_rate"],
        "deployed_threshold_within_swept_range": 0.0 <= threshold <= 1.0,
        "deployed_threshold_not_tuned_on_test":
            abs(threshold - results["val"]["best_f1_threshold_on_this_split"]) < 1e-6,
    }
    status = verdict(all(checks.values()))
    print()
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 6: {status}")

    result = {
        "task": 6, "name": "Threshold sweep (ROC / PR)", "status": status,
        "coarse_step": COARSE_STEP, "fine_step": FINE_STEP,
        "deployed_threshold": threshold,
        "splits": results,
        "checks": checks,
    }
    write_csv(coarse_rows, reports_dir(config) / "threshold_sweep.csv")
    result["roc_figure"] = _plot_roc(curves, results, threshold, config)
    result["pr_figure"] = _plot_pr(curves, results, threshold, config)
    write_json(result, reports_dir(config) / "threshold_sweep.json")
    return result


def _nearest(rows, threshold):
    """Return the swept record closest to a given threshold.

    Args:
        rows: Sweep records.
        threshold: Threshold to locate.

    Returns:
        dict: The nearest record.
    """
    return min(rows, key=lambda r: abs(r["threshold"] - threshold))


def _plot_roc(curves, results, threshold, config):
    """Plot ROC curves for both splits, marking the deployed threshold.

    Args:
        curves: Fine sweep records per split.
        results: Per-split summaries, read for the areas shown in the legend.
        threshold: Deployed threshold, drawn as a marker.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    new_figure()
    for name, color in (("val", COLORS["val"]), ("test", COLORS["test"])):
        rows = sorted(curves[name], key=lambda r: r["FPR"])
        plt.plot([r["FPR"] for r in rows], [r["TPR_recall"] for r in rows], color=color,
                 linewidth=2, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
        point = _nearest(curves[name], threshold)
        plt.plot(point["FPR"], point["TPR_recall"], "o", color=color, markersize=8,
                 markeredgecolor="white")
    plt.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="--", linewidth=1,
             label="No skill")
    plt.title(f"ROC curve (marker = deployed threshold {threshold:.2f})")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    return finish(graphs_dir(config) / "roc_curve.png")


def _plot_pr(curves, results, threshold, config):
    """Plot precision-recall curves for both splits, marking the deployed threshold.

    Each split's no-skill baseline, equal to its positive rate, is drawn for reference.

    Args:
        curves: Fine sweep records per split.
        results: Per-split summaries, read for the areas shown in the legend.
        threshold: Deployed threshold, drawn as a marker.
        config: Project configuration.

    Returns:
        str: Path to the written figure.
    """
    new_figure()
    for name, color in (("val", COLORS["val"]), ("test", COLORS["test"])):
        rows = sorted(curves[name], key=lambda r: r["TPR_recall"])
        plt.plot([r["TPR_recall"] for r in rows], [r["precision"] for r in rows], color=color,
                 linewidth=2, label=f"{name} (AP = {results[name]['average_precision']:.3f})")
        point = _nearest(curves[name], threshold)
        plt.plot(point["TPR_recall"], point["precision"], "o", color=color, markersize=8,
                 markeredgecolor="white")
        plt.axhline(results[name]["positive_rate"], color=color, linestyle=":", linewidth=1,
                    alpha=0.7, label=f"{name} no-skill ({results[name]['positive_rate']:.3f})")
    plt.title(f"Precision-Recall curve (marker = deployed threshold {threshold:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    return finish(graphs_dir(config) / "pr_curve.png")


if __name__ == "__main__":
    run()
