"""Task 5: core statistical metrics on the held-out test split.

Reports the confusion matrix and derived metrics for recordings never seen in training
and never used to tune the detection threshold.

Accuracy alone is not informative on this corpus, which is predominantly ambience: a
detector that never fires already scores well above chance. Precision, recall, F1 and
metrics that account for class imbalance are reported instead, alongside the confusion
matrix itself so any other figure can be recomputed.

Results are given at two operating points, the neutral 0.5 and the deployed threshold.

The per-location breakdown accompanies the headline figures because several recording
locations contain only one class. At such a location the label follows from the
environment, so an aggregate metric can be high without any detection taking place. The
breakdown identifies which locations support a meaningful measurement.

Run from the repository root::

    python -m aerosonar.evaluation.metrics
"""
import math

import torch

from aerosonar.config import load_default_config
from aerosonar.data.dataset import (METADATA_PATH, TENSOR_DIR, SpectrogramTensorDataset,
                                    load_joined_metadata, split_file_ids)
from aerosonar.evaluation.common import (graphs_dir, load_deployed_threshold,
                                         load_trained_model, reports_dir, resolve_device,
                                         section, verdict, write_csv, write_json)
from aerosonar.training.trainCNN import confusion_metrics
from aerosonar.utils.plotting import plot_confusion_matrix

BATCH_SIZE = 128


def score_split(config, model, device, file_ids, meta):
    """Score every chunk of a split.

    Args:
        config: Project configuration.
        model: Trained model in evaluation mode.
        device: Compute device.
        file_ids: Recordings comprising the split.
        meta: Joined metadata with ``filename``, ``target``, ``file_id`` and
            ``location`` columns.

    Returns:
        pandas.DataFrame: The split's metadata rows with an added ``prob`` column
        holding the drone probability.
    """
    rows = meta[meta["file_id"].isin(file_ids)].reset_index(drop=True)
    dataset = SpectrogramTensorDataset(METADATA_PATH, TENSOR_DIR)
    lookup = {name: i for i, name in enumerate(dataset.metadata["filename"])}

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            chunk = rows.iloc[start:start + BATCH_SIZE]
            batch = torch.stack([dataset[lookup[name]][0] for name in chunk["filename"]])
            logits = model(batch.to(device).float())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    rows = rows.copy()
    rows["prob"] = probabilities
    return rows


def confusion_at(rows, threshold):
    """Compute the confusion matrix at one threshold.

    Args:
        rows: Scored rows with ``prob`` and ``target`` columns.
        threshold: Probability above which a chunk is called a drone.

    Returns:
        tuple: ``(TP, FP, TN, FN)``.
    """
    predictions = (rows["prob"] > threshold).astype(int)
    actual = rows["target"].astype(int)
    return (
        int(((predictions == 1) & (actual == 1)).sum()),  # TP
        int(((predictions == 1) & (actual == 0)).sum()),  # FP
        int(((predictions == 0) & (actual == 0)).sum()),  # TN
        int(((predictions == 0) & (actual == 1)).sum()),  # FN
    )


def full_metrics(TP, FP, TN, FN):
    """Derive the full metric set from a confusion matrix.

    Specificity, balanced accuracy and the Matthews correlation coefficient are included
    because they expose a detector that merely predicts the majority class. Such a model
    can post a high plain accuracy while its MCC remains near zero.

    Args:
        TP: True positive count.
        FP: False positive count.
        TN: True negative count.
        FN: False negative count.

    Returns:
        dict: The four counts, per-class support, accuracy, precision, recall, F1,
        specificity, balanced accuracy, false positive and negative rates, and MCC.
    """
    accuracy, precision, recall, f1 = confusion_metrics(TP, FP, TN, FN)
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "support_drone": TP + FN, "support_ambience": TN + FP,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": FP / (FP + TN) if (FP + TN) else 0.0,
        "false_negative_rate": FN / (FN + TP) if (FN + TP) else 0.0,
        "mcc": ((TP * TN - FP * FN) / denominator) if denominator else 0.0,
    }


def _print_metrics(label, m):
    """Print one metric set as an indented block.

    Args:
        label: Heading for the block.
        m: Metric dictionary from :func:`full_metrics`.
    """
    print(f"\n  {label}")
    print(f"    TP={m['TP']:5d}  FP={m['FP']:5d}  TN={m['TN']:5d}  FN={m['FN']:5d}")
    print(f"    accuracy  {m['accuracy']:.4f}   precision {m['precision']:.4f}   "
          f"recall {m['recall']:.4f}   F1 {m['f1']:.4f}")
    print(f"    specificity {m['specificity']:.4f}   balanced acc {m['balanced_accuracy']:.4f}   "
          f"MCC {m['mcc']:.4f}")
    print(f"    FPR {m['false_positive_rate']:.4f}   FNR {m['false_negative_rate']:.4f}")


def run(config=None):
    """Evaluate the trained model on the held-out test split.

    Writes ``confusion_matrix_test.csv``, ``per_location_metrics.csv``,
    ``chunk_probabilities.csv``, ``test_metrics.json`` and
    ``confusion_matrix_test.png``.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Result record with ``status``, both operating points, the per-location
        breakdown, metrics pooled over locations containing both classes, and the
        individual check outcomes.
    """
    config = config or load_default_config()
    device = resolve_device(config)
    section("TASK 5 — CORE STATISTICAL METRICS (held-out test split)")

    model = load_trained_model(config, device)
    threshold = load_deployed_threshold(config)
    meta = load_joined_metadata()
    train_ids, val_ids, test_ids = split_file_ids(meta)

    scored = {
        "val": score_split(config, model, device, val_ids, meta),
        "test": score_split(config, model, device, test_ids, meta),
    }
    rows = scored["test"]

    n_drone = int((rows["target"] == 1).sum())
    n_ambience = int((rows["target"] == 0).sum())
    print(f"Test split: {len(rows)} chunks from {rows['file_id'].nunique()} recordings "
          f"({n_drone} drone, {n_ambience} ambience)")
    print(f"Class balance: {n_ambience / len(rows):.1%} ambience — a detector that never "
          f"fires would already score {n_ambience / len(rows):.1%} 'accuracy', which is why "
          f"precision/recall/F1 are the reported figures.")
    print(f"Deployed threshold: {threshold:.2f} (tuned on validation, not on test)")

    operating_points = {}
    for name, value in (("threshold_0.50", 0.5), (f"threshold_{threshold:.2f}_deployed", threshold)):
        metrics = full_metrics(*confusion_at(rows, value))
        metrics["threshold"] = value
        operating_points[name] = metrics
        _print_metrics(f"@ {name}", metrics)

    print("\n  Per-location breakdown on test (deployed threshold):")
    location_rows = []
    for location, group in rows.groupby("location"):
        metrics = full_metrics(*confusion_at(group, threshold))
        classes = sorted(group["target"].unique().tolist())
        entry = {
            "location": location, "chunks": len(group),
            "recordings": int(group["file_id"].nunique()),
            "classes_present": "both" if len(classes) == 2 else ("drone" if classes == [1] else "ambience"),
            **{k: v for k, v in metrics.items()},
        }
        location_rows.append(entry)
        note = "" if len(classes) == 2 else "  <- single-class: only one error type is defined here"
        print(f"    {location:22s} n={len(group):5d} [{entry['classes_present']:9s}] "
              f"acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
              f"rec={metrics['recall']:.3f} f1={metrics['f1']:.3f}{note}")

    both_class_locations = [r for r in location_rows if r["classes_present"] == "both"]
    if both_class_locations:
        pooled = full_metrics(
            *(sum(r[k] for r in both_class_locations) for k in ("TP", "FP", "TN", "FN"))
        )
        print(f"\n  Pooled over locations that contain BOTH classes "
              f"({', '.join(r['location'] for r in both_class_locations)}) — the subset where "
              f"the score cannot be explained by recognising the location:")
        _print_metrics("both-class locations only", pooled)
    else:
        pooled = None
        print("\n  WARNING: no test location contains both classes, so every number above is "
              "confounded with location identity. See HANDOFF.md.")

    deployed = operating_points[f"threshold_{threshold:.2f}_deployed"]
    checks = {
        "test_split_nonempty": len(rows) > 0,
        "both_classes_present_in_test": n_drone > 0 and n_ambience > 0,
        "detector_actually_fires": deployed["TP"] + deployed["FP"] > 0,
        "beats_majority_class_baseline":
            deployed["accuracy"] > max(n_drone, n_ambience) / len(rows),
        # Aggregate metrics can be carried entirely by single-class locations, where
        # identifying the environment yields the correct label without any detection.
        # Only a location holding both classes tests drone against no-drone, so a
        # near-zero MCC there indicates no discriminative power whatever the headline
        # accuracy shows.
        "discriminates_within_a_single_location":
            pooled is not None and pooled["mcc"] > 0.1,
    }
    status = verdict(all(checks.values()))
    print()
    for name, ok in checks.items():
        print(f"  [{verdict(ok)}] {name}")
    print(f"\nTask 5: {status}")

    result = {
        "task": 5, "name": "Core statistical metrics", "status": status,
        "split": "test",
        "chunks": len(rows),
        "recordings": int(rows["file_id"].nunique()),
        "support_drone": n_drone,
        "support_ambience": n_ambience,
        "majority_class_baseline": max(n_drone, n_ambience) / len(rows),
        "deployed_threshold": threshold,
        "operating_points": operating_points,
        "per_location": location_rows,
        "pooled_both_class_locations": pooled,
        "checks": checks,
    }

    write_csv([
        {"threshold": m["threshold"], **{k: v for k, v in m.items() if k != "threshold"}}
        for m in operating_points.values()
    ], reports_dir(config) / "confusion_matrix_test.csv")
    write_csv(location_rows, reports_dir(config) / "per_location_metrics.csv")
    write_csv(
        [{"split": name, **row} for name, frame in scored.items()
         for row in frame[["filename", "file_id", "location", "target", "prob"]].to_dict("records")],
        reports_dir(config) / "chunk_probabilities.csv",
    )
    result["figure"] = plot_confusion_matrix(
        deployed["TP"], deployed["FP"], deployed["TN"], deployed["FN"],
        f"Test Confusion Matrix (threshold={threshold:.2f})",
        graphs_dir(config) / "confusion_matrix_test.png",
    )
    write_json(result, reports_dir(config) / "test_metrics.json")
    return result


if __name__ == "__main__":
    run()
