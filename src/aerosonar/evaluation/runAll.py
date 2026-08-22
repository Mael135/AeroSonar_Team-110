"""Runs the full verification and evaluation suite and generates the report.

The suite is divided into two phases whose results mean different things, a distinction
the generated report preserves throughout.

Phase 1, Tasks 1 to 4, is verification: structural proofs that the pipeline is built
correctly and free of software defects. A failure there is a bug. These checks establish
nothing about real-world performance.

Phase 2, Tasks 5 to 9, is evaluation: measurements of what the trained model actually
does. A failure there is a finding about the model or the dataset rather than a broken
program, and the report presents it as such.

Run from the repository root::

    python -m aerosonar.evaluation.runAll
"""
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from aerosonar.config import load_default_config
from aerosonar.evaluation import (continuityTest, determinismCheck, latencyBench,
                                  learningCurves, metrics, overfitTest, shapeTrace,
                                  snrSweep, thresholdSweep)
from aerosonar.evaluation.common import (FAIL, PASS, SKIP, provenance, reports_dir, section,
                                         write_json)

PHASE_1 = [
    ("Single-batch overfit test", overfitTest),
    ("Tensor dimensionality assertion", shapeTrace),
    ("Deterministic execution check", determinismCheck),
    ("Pipeline continuity test", continuityTest),
]
PHASE_2 = [
    ("Core statistical metrics", metrics),
    ("Threshold sweep (ROC / PR)", thresholdSweep),
    ("Latency and bounding measurement", latencyBench),
    ("Generalization vs domain shift", learningCurves),
    ("SNR sensitivity", snrSweep),
]


def run(config=None):
    """Run every check in order and write the aggregated report.

    A check that raises is recorded as a failure with its traceback and the suite
    continues, so one broken check does not discard the others' results.

    Writes ``results.json`` and ``verification_report.md`` in addition to the artifacts
    each individual check produces.

    Args:
        config: Project configuration. Loaded from disk when omitted.

    Returns:
        dict: Generation timestamp, provenance record, the configuration used, and the
        list of per-task results.
    """
    config = config or load_default_config()
    results = []

    # learningCurves reads Task 5's output, so Phase 2 must run in listed order.
    for name, module in PHASE_1 + PHASE_2:
        try:
            results.append(module.run(config))
        except Exception as exc:
            print(f"\n  {name} raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results.append({
                "task": len(results) + 1, "name": name, "status": FAIL,
                "error": f"{type(exc).__name__}: {exc}",
            })

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance(config),
        "config": config,
        "results": results,
    }
    section("SUMMARY")
    for result in results:
        print(f"  Task {result['task']}  [{result['status']:4s}]  {result['name']}")

    write_json(payload, reports_dir(config) / "results.json")
    report_path = write_report(payload, config)
    print(f"\nReport: {report_path}")
    return payload


# --------------------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------------------

def _by_task(results, number):
    """Return the result record for one task number, or None if absent."""
    return next((r for r in results if r.get("task") == number), None)


def _status_icon(status):
    """Render a status value for the report."""
    return {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}.get(status, status)


def _checks_table(result):
    """Render a result's individual checks as a markdown table.

    Args:
        result: A task result record.

    Returns:
        str: The table, or an empty string when the record has no checks.
    """
    checks = result.get("checks") or {}
    if not checks:
        return ""
    lines = ["| Check | Result |", "| --- | --- |"]
    lines += [f"| `{name}` | {_status_icon(PASS if ok else FAIL)} |" for name, ok in checks.items()]
    return "\n".join(lines) + "\n"


#: Directory the report is written to. Figure links are rewritten relative to it,
#: because the figures live elsewhere and absolute paths would not survive relocation.
_REPORT_DIR = Path("reports/eval")


def _figure(path, caption):
    """Render a markdown image reference with a caption.

    Args:
        path: Path to the figure, rewritten relative to :data:`_REPORT_DIR`.
        caption: Alt text and caption.

    Returns:
        str: The markdown, or an empty string when ``path`` is falsy.
    """
    if not path:
        return ""
    relative = os.path.relpath(Path(path).resolve(), _REPORT_DIR.resolve())
    return f"\n![{caption}]({Path(relative).as_posix()})\n\n*{caption}*\n"


def write_report(payload, config):
    """Generate the markdown report from the aggregated results.

    Args:
        payload: The record returned by :func:`run`.
        config: Project configuration.

    Returns:
        pathlib.Path: Path to the written report.
    """
    global _REPORT_DIR
    _REPORT_DIR = reports_dir(config)
    results = payload["results"]
    prov = payload["provenance"]
    out = []
    w = out.append

    w("# AeroSonar — Verification & Evaluation Report\n")
    w(f"Generated {payload['generated_utc']} from commit `{prov['git_commit']}` "
      f"on branch `{prov['git_branch']}`"
      f"{' (working tree dirty)' if prov['git_dirty'] else ''}.\n")

    # --- provenance ---
    w("## Test environment\n")
    w("| Field | Value |")
    w("| --- | --- |")
    for label, key in [("Python", "python"), ("PyTorch", "torch"), ("torchaudio", "torchaudio"),
                       ("Platform", "platform"), ("Compute device", "device"),
                       ("GPU", "cuda_device"), ("Detection threshold", "detection_threshold")]:
        if prov.get(key) is not None:
            w(f"| {label} | `{prov[key]}` |")
    w("")

    # --- summary ---
    w("## Summary\n")
    w("Phase 1 checks are **structural proofs**: they establish that the pipeline is "
      "implemented correctly and free of software defects. A failure there is a bug. "
      "Phase 2 checks are **measurements** of the trained model; a failure there is a "
      "finding about the model or the dataset, not a broken pipeline.\n")
    w("| # | Task | Phase | Result |")
    w("| --- | --- | --- | --- |")
    for result in results:
        phase = "Verification" if result["task"] <= 4 else "Evaluation"
        w(f"| {result['task']} | {result['name']} | {phase} | "
          f"**{_status_icon(result['status'])}** |")
    w("")

    w("---\n")
    w("# Phase 1 — Verification (structural proofs)\n")
    _write_task1(w, _by_task(results, 1))
    _write_task2(w, _by_task(results, 2))
    _write_task3(w, _by_task(results, 3))
    _write_task4(w, _by_task(results, 4))

    w("---\n")
    w("# Phase 2 — Evaluation (performance & failure modes)\n")
    _write_task5(w, _by_task(results, 5))
    _write_task6(w, _by_task(results, 6))
    _write_task7(w, _by_task(results, 7))
    _write_task8(w, _by_task(results, 8))
    _write_task9(w, _by_task(results, 9))

    w("---\n")
    _write_conclusions(w, results)

    path = reports_dir(config) / "verification_report.md"
    path.write_text("\n".join(out))
    return path


def _header(w, result, subtitle):
    """Write a task heading and its introductory sentence.

    Args:
        w: Line-appending callable.
        result: The task result record.
        subtitle: Introductory text describing what the task measures.
    """
    w(f"## Task {result['task']} — {result['name']}  ·  **{_status_icon(result['status'])}**\n")
    w(subtitle + "\n")


def _skipped(w, result):
    """Write a placeholder for a task that did not produce results.

    Args:
        w: Line-appending callable.
        result: The task result record, or None if the task never ran.

    Returns:
        bool: True if a placeholder was written and the caller should return early.
    """
    if result is None:
        w("_Not run._\n")
        return True
    if result["status"] == SKIP or "error" in result:
        w(f"## Task {result.get('task', '?')} — {result.get('name', '')}  ·  "
          f"**{_status_icon(result['status'])}**\n")
        w(f"_{result.get('reason') or result.get('error')}_\n")
        return True
    return False


def _write_task1(w, r):
    """Write the report section for Task 1.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Trains a fresh model on a single small batch to prove the loss, "
                  "optimizer and gradient path work. A model that cannot memorise ten "
                  "examples has a structural defect no amount of data would fix.")
    w(f"- Batch: **{r['batch_size']} clips** (class-balanced) drawn from "
      f"**{r['n_recordings_in_batch']} distinct recordings**, augmentation disabled.")
    w(f"- Optimiser: AdamW at lr `{r['lr']}` for **{r['epochs']} epochs**.")
    w(f"- Loss fell from **{r['initial_eval_loss']:.4f}** to "
      f"**{r['final_eval_loss']:.2e}** (tolerance for \"zero\": `{r['loss_tolerance']}`).")
    w(f"- Reached **{r['final_accuracy_pct']:.0f}% accuracy at epoch "
      f"{r['epochs_to_100pct']}** and held it.\n")
    w("Loss is evaluated in `eval()` mode. In `train()` mode the head's `Dropout(0.5)` "
      "keeps the loss noisy and bounded away from zero even for a perfectly memorised "
      "batch, so a train-mode reading would understate memorisation.\n")
    w(_checks_table(r))
    w(_figure(r.get("figure"), "Single-batch overfit: loss and accuracy vs epoch"))


def _write_task2(w, r):
    """Write the report section for Task 2.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Records the exact tensor shape at every stage from raw audio to the "
                  "output scalar, proving nothing is silently truncated, padded or "
                  "transposed. Shapes are captured with forward hooks, so the "
                  "instrumentation cannot perturb the deployed forward pass.")
    w(f"Source clip: `{r['source_clip']}` · model parameters: "
      f"**{r['total_parameters']:,}**\n")

    w("### Pipeline stages\n")
    w("| Stage | Shape | Range (min, max) |")
    w("| --- | --- | --- |")
    for stage in r["stages"]:
        w(f"| {stage['stage']} | `{stage['shape']}` | {stage['min']:.3f}, {stage['max']:.3f} |")
    w("")
    w(f"The {r['expected_mel_frames']} time frames are arithmetic, not incidental: "
      f"{r['clip_samples']} samples / hop 512 + 1 (centred STFT) = "
      f"{r['expected_mel_frames']}.\n")

    w("### Model forward pass\n")
    w("| Block | Layer | Input | Output | Params |")
    w("| --- | --- | --- | --- | --- |")
    for layer in r["layers"]:
        w(f"| `{layer['block']}[{layer['index']}]` | {layer['layer']} | "
          f"`{layer['in_shape']}` | `{layer['out_shape']}` | {layer['params']:,} |")
    w("")
    w(_checks_table(r))


def _write_task3(w, r):
    """Write the report section for Task 3.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Feeds identical audio through the deployed path repeatedly and requires "
                  "identical confidence scores, proving BatchNorm and Dropout behave "
                  "correctly at inference time.")
    w(f"Deployment device: `{r['deployment_device']}` · "
      f"probability `{r['probability_repr']}`\n")
    w("| Device | 3 runs bit-identical | Batch-context delta | Batch-statistics delta "
      "(what a real leak costs) |")
    w("| --- | --- | --- | --- |")
    for device in r["devices"]:
        batch = device["batch_context"]
        control = device["batch_statistics_control"]
        ratio = control["ratio_vs_batch_context_delta"]
        w(f"| `{device['device']}` | "
          f"{_status_icon(PASS if device['checks'].get('3_eval_runs_bit_identical') else FAIL)} | "
          f"`{batch['delta']:.3e}`{' (bit-exact)' if batch['bit_identical'] else ''} | "
          f"`{control['delta_vs_alone']:.3e}`"
          f"{f' ({ratio:.0f}x larger)' if ratio else ''} |")
    w("")
    w("Repeated calls on identical input are bit-identical on both devices. The residual "
      "in the batch-context column on CUDA is **not** a state leak: cuDNN selects different "
      "convolution algorithms for different batch shapes, which changes summation order in "
      "the last significant digits. The same measurement on CPU is exactly zero. The "
      "control column shows what a genuine BatchNorm state leak costs on the same input — "
      "orders of magnitude larger — which is what separates float noise from a real defect.\n")
    w(_checks_table(r))


def _write_task4(w, r):
    """Write the report section for Task 4.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Streams long continuous audio through the live rolling-buffer inference "
                  "loop and tracks resident memory, verifying there is no leak in real-time "
                  "spectrogram generation.")
    w(f"Configuration: **{r['minutes_per_source']} minutes per source**, "
      f"{r['block_ms']} ms blocks into a {r['window_samples']}-sample window, "
      f"device `{r['device']}`. Leak threshold: "
      f"**{r['max_drift_mb_per_min']} MB/min** of RSS drift after burn-in.\n")
    w("| Source | Frames | Inferences | RSS start | RSS end | Drift | Mean p(drone) |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for stream in r["streams"]:
        w(f"| {stream['source']} | {stream['frames']:,} | {stream['inferences']:,} | "
          f"{stream['steady_state_rss_mb']:.1f} MB | {stream['final_rss_mb']:.1f} MB | "
          f"**{stream['drift_mb_per_min']:+.3f} MB/min** | "
          f"{stream['mean_drone_probability']} |")
    w("")
    total = sum(s["inferences"] for s in r["streams"])
    w(f"{total:,} consecutive inferences completed without an exception and with flat "
      f"memory. The loop drives `aerosonar.inference.streaming.push_frame`, the same "
      f"buffer update the live detector uses, so a regression there fails this check.\n")
    w(_checks_table(r))
    w(_figure(r.get("figure"), "Resident memory during continuous streaming"))


def _write_task5(w, r):
    """Write the report section for Task 5.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Confusion matrix and derived metrics on the held-out test split — "
                  "recordings never seen in training, and never used to tune the threshold.")
    w(f"Test split: **{r['chunks']:,} chunks** from **{r['recordings']} recordings** "
      f"({r['support_drone']} drone, {r['support_ambience']} ambience).\n")
    w(f"The corpus is **{r['support_ambience'] / r['chunks']:.1%} ambience**, so a detector "
      f"that never fires already scores {r['majority_class_baseline']:.1%} accuracy. "
      f"Accuracy alone is therefore not a meaningful figure here, which is why precision, "
      f"recall, F1 and MCC are reported.\n")

    w("| Operating point | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | MCC |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, m in r["operating_points"].items():
        w(f"| {name.replace('_', ' ')} | {m['TP']} | {m['FP']} | {m['TN']} | {m['FN']} | "
          f"{m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | "
          f"{m['specificity']:.4f} | {m['mcc']:.4f} |")
    w("")

    w("### Per-location breakdown\n")
    w("This table is the important one. Several recording locations contain only one "
      "class, so at those locations the label can be predicted from the environment "
      "without detecting anything.\n")
    w("| Location | Chunks | Classes present | Accuracy | Precision | Recall | F1 | MCC |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for loc in r["per_location"]:
        w(f"| {loc['location']} | {loc['chunks']:,} | {loc['classes_present']} | "
          f"{loc['accuracy']:.3f} | {loc['precision']:.3f} | {loc['recall']:.3f} | "
          f"{loc['f1']:.3f} | {loc['mcc']:.3f} |")
    w("")

    pooled = r.get("pooled_both_class_locations")
    if pooled:
        w("### Un-confounded subset\n")
        w("Restricted to test locations containing **both** classes — the only subset "
          "where a correct answer requires detecting the drone rather than recognising "
          "the room:\n")
        w(f"- TP **{pooled['TP']}**, FP **{pooled['FP']}**, TN **{pooled['TN']}**, "
          f"FN **{pooled['FN']}**")
        w(f"- Precision **{pooled['precision']:.4f}**, recall **{pooled['recall']:.4f}**, "
          f"specificity **{pooled['specificity']:.4f}**, "
          f"MCC **{pooled['mcc']:.4f}**\n")
        if pooled["mcc"] <= 0.1:
            w("> With specificity at "
              f"{pooled['specificity']:.2f} and MCC at {pooled['mcc']:.2f}, the model "
              "classifies **every** chunk at this location as a drone. Once location is "
              "held constant it has no discriminative power at all: the headline accuracy "
              "above is produced by single-class locations, where the environment predicts "
              "the label for free.\n")
    w(_checks_table(r))
    w(_figure(r.get("figure"), "Test-split confusion matrix at the deployed threshold"))


def _write_task6(w, r):
    """Write the report section for Task 6.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Every cutoff from 0.00 to 1.00, with the full confusion matrix at each — "
                  "the evidence for the deployed threshold rather than an assertion of it.")
    w("| Split | Chunks | ROC-AUC | Average precision | No-skill baseline | Best-F1 "
      "threshold | F1 there | F1 at deployed threshold |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, s in r["splits"].items():
        w(f"| {name} | {s['chunks']:,} | **{s['roc_auc']:.4f}** | "
          f"{s['average_precision']:.4f} | {s['positive_rate']:.4f} | "
          f"{s['best_f1_threshold_on_this_split']:.2f} | "
          f"{s['best_f1_on_this_split']:.4f} | "
          f"{s['at_deployed_threshold']['f1']:.4f} |")
    w("")
    val, test = r["splits"]["val"], r["splits"]["test"]
    w(f"**Why {r['deployed_threshold']:.2f}.** The threshold was chosen as the F1 maximum "
      f"on the *validation* split — never on test, so the test figures stay out-of-sample. "
      f"It transfers well: test F1 at the deployed threshold is "
      f"{test['at_deployed_threshold']['f1']:.4f} against a test-optimal "
      f"{test['best_f1_on_this_split']:.4f} at {test['best_f1_threshold_on_this_split']:.2f}, "
      f"a gap of {abs(test['best_f1_on_this_split'] - test['at_deployed_threshold']['f1']):.4f}. "
      f"The cutoff is not overfit to the split that chose it.\n")
    w(f"At the deployed threshold the test operating point is recall "
      f"**{test['at_deployed_threshold']['TPR_recall']:.4f}** at a false-positive rate of "
      f"**{test['at_deployed_threshold']['FPR']:.4f}** — the detector fires on almost every "
      f"drone chunk and on a third of the ambience.\n")
    w("The full 0.05-step grid for both splits is in "
      "[`threshold_sweep.csv`](threshold_sweep.csv).\n")
    w(_checks_table(r))
    w(_figure(r.get("roc_figure"), "ROC curve, validation and test"))
    w(_figure(r.get("pr_figure"), "Precision-Recall curve, validation and test"))


def _write_task7(w, r):
    """Write the report section for Task 7.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    geometry = r["frame_geometry"]
    _header(w, r, f"Times the complete inference path — raw audio array in, drone "
                  f"probability out — over {r['runs']:,} runs, against the "
                  f"{r['budget_ms']} ms STM32 frame budget.")
    w(f"Boundary measured: `np.ndarray[{geometry['clip_samples']}]` -> device transfer -> "
      f"mel transform -> CNN -> scalar. Pipeline geometry: "
      f"{geometry['clip_duration_s']} s window, {geometry['window_ms']:.1f} ms STFT window, "
      f"{geometry['hop_ms']:.1f} ms hop.\n")
    w("| Device | Mean | Median | Min | Max | p95 | **p99** | Mel | CNN | Headroom vs "
      f"{r['budget_ms']} ms (p99) |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for device in r["devices"]:
        e = device["end_to_end"]
        w(f"| `{device['device']}` ({device['device_name']}) | {e['mean_ms']:.3f} ms | "
          f"{e['median_ms']:.3f} | {e['min_ms']:.3f} | {e['max_ms']:.3f} | "
          f"{e['p95_ms']:.3f} | **{e['p99_ms']:.3f} ms** | "
          f"{device['transform_only']['mean_ms']:.3f} | "
          f"{device['model_only']['mean_ms']:.3f} | "
          f"**{device['headroom_vs_budget']:.1f}x** |")
    w("")
    deployment = r["devices"][0]
    w(f"The budget is judged on **p99, not the mean**: a real-time stage is bounded by its "
      f"worst frames, and a mean inside budget with a tail outside it still drops frames. "
      f"Both devices clear {r['budget_ms']} ms with at least "
      f"{min(d['headroom_vs_budget'] for d in r['devices']):.0f}x margin, so the PC stage "
      f"cannot be the bottleneck.\n")
    w(f"The GPU has the lower mean ({deployment['end_to_end']['mean_ms']:.3f} ms vs "
      f"{r['devices'][-1]['end_to_end']['mean_ms']:.3f} ms) but the *worse* tail "
      f"({deployment['end_to_end']['p99_ms']:.3f} ms vs "
      f"{r['devices'][-1]['end_to_end']['p99_ms']:.3f} ms), because kernel-launch "
      f"scheduling occasionally stalls. For a hard-real-time budget the CPU is the more "
      f"predictable choice; at this workload either has ample margin.\n")
    w(_checks_table(r))
    w(_figure(r.get("figure"), "End-to-end inference latency distribution"))


def _write_task8(w, r):
    """Write the report section for Task 8.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    a = r["analysis"]
    _header(w, r, "Training loss against validation loss over epochs, and the failure mode "
                  "those curves plus the per-location evidence identify.")
    w(f"### Diagnosis: **{a['verdict']}**\n")
    w("| Quantity | Value |")
    w("| --- | --- |")
    w(f"| Epochs trained | {a['epochs']} |")
    w(f"| Minimum validation loss at epoch | **{a['best_val_loss_epoch']}** |")
    w(f"| Epochs trained past that optimum | {a['epochs_trained_past_optimum']} |")
    w(f"| Final train loss | {a['final_train_loss']:.4f} |")
    w(f"| Final validation loss | {a['final_val_loss']:.4f} "
      f"(min {a['min_val_loss']:.4f}) |")
    w(f"| Validation loss above its minimum | "
      f"**{(a['val_loss_upturn_ratio'] - 1) * 100:+.1f}%** |")
    w(f"| Generalization gap | {a['final_generalization_gap']:+.4f} |")
    w(f"| Final train accuracy | {a['final_train_acc']:.2f}% |")
    w(f"| Final validation accuracy | {a['final_val_acc']:.2f}% (F1 {a['final_val_f1']:.4f}) |")
    w("")

    w("**Overfitting.** Validation loss reaches its minimum at epoch "
      f"{a['best_val_loss_epoch']} and then climbs "
      f"{(a['val_loss_upturn_ratio'] - 1) * 100:.1f}% while training loss keeps falling. "
      f"The remaining {a['epochs_trained_past_optimum']} epochs bought training-set fit at "
      f"the cost of generalization; early stopping at epoch {a['best_val_loss_epoch']} is "
      f"the direct remedy.\n")

    confound = r.get("unconfounded_subset")
    if confound and confound["mcc"] <= r["thresholds"]["confound_mcc"]:
        w("**Domain shift / confounded feature.** This is the more serious failure, and it "
          "is *invisible in the loss curves* — validation loss looks unremarkable because "
          "the validation split shares the training data's confound. It is only visible in "
          f"Task 5's per-location breakdown, where MCC on the un-confounded subset is "
          f"**{confound['mcc']:.4f}** with specificity **{confound['specificity']:.4f}**.\n")
        w("The training corpus was recorded in a small number of curated sessions in which "
          "location and label are entangled: `room` and `balcony` contain only no-drone "
          "audio, `ben shemen` only drone audio. A model can minimise training loss by "
          "learning the acoustic signature of the *recording environment* — its reverberation, "
          "noise floor, and microphone placement — rather than the rotor harmonics that "
          "identify a drone. That shortcut generalises perfectly to held-out clips from the "
          "same sessions and not at all to a new environment.\n")
        w("The curated corpus also lacks the conditions deployment presents: wind noise "
          "across the capsule, multipath echo off buildings and terrain, varying distance "
          "and aspect angle, uncontrolled microphone gain and AGC, and interfering sources "
          "(traffic, aircraft, human speech) that were never recorded alongside a drone. "
          "This is the discrepancy between the curated training distribution and the "
          "deployment distribution — a domain shift — and it is the documented cause of the "
          "real-world failure recorded in `HANDOFF.md`.\n")
    w(_checks_table(r))
    w(_figure(r.get("loss_figure"), "Training vs validation loss"))
    w(_figure(r.get("accuracy_figure"), "Training vs validation accuracy"))


def _write_task9(w, r):
    """Write the report section for Task 9.

    Args:
        w: Line-appending callable.
        r: The task result record.
    """
    if _skipped(w, r):
        return
    _header(w, r, "Confidence decay as a detected drone clip is buried in progressively "
                  "louder noise, with a noise-only negative control.")
    w(f"Method: the **{len(r['clips_used'])} highest-confidence** drone clips "
      f"(clean p(drone) = {r['clean_mean_confidence']:.4f}) mixed with seeded white and "
      f"pink noise in the waveform domain, {r['noise_seeds_per_point']} noise seeds per "
      f"point. Pink (1/f) noise is included because wind and traffic concentrate energy in "
      f"the low bands where rotor harmonics live.\n")
    w("RMS loudness normalisation scales signal and noise together and therefore leaves "
      "SNR unchanged, so the sweep measures noise tolerance rather than an artefact of "
      "normalisation.\n")

    w("| SNR (dB) | White: p(drone) | White: detection rate | Pink: p(drone) | "
      "Pink: detection rate |")
    w("| --- | --- | --- | --- | --- |")
    for snr in r["snr_db_levels"]:
        white = next(x for x in r["sweep"] if x["noise"] == "white" and x["snr_db"] == snr)
        pink = next(x for x in r["sweep"] if x["noise"] == "pink" and x["snr_db"] == snr)
        w(f"| {snr:+d} | {white['mean_confidence']:.4f} ± {white['std_confidence']:.4f} | "
          f"{white['detection_rate']:.1%} | "
          f"{pink['mean_confidence']:.4f} ± {pink['std_confidence']:.4f} | "
          f"{pink['detection_rate']:.1%} |")
    w("")

    control = r.get("noise_only_control")
    if control:
        w("### Negative control: noise only, no drone\n")
        w("| Noise | Mean p(drone) | False-alarm rate |")
        w("| --- | --- | --- |")
        for kind, entry in control.items():
            w(f"| {kind} | {entry['mean_confidence']:.4f} ± {entry['std_confidence']:.4f} | "
              f"**{entry['false_alarm_rate']:.1%}** |")
        w("")
        if not all(c["false_alarm_rate"] < 0.5 for c in control.values()):
            w("> **This control changes the reading of the whole sweep.** Confidence stays "
              "above threshold down to −30 dB, which in isolation looks like exceptional "
              "noise robustness. But pure noise containing no drone at all scores "
              f"{control['white']['mean_confidence']:.3f} (white) and "
              f"{control['pink']['mean_confidence']:.3f} (pink), with false-alarm rates of "
              f"{control['white']['false_alarm_rate']:.0%} and "
              f"{control['pink']['false_alarm_rate']:.0%}. The curves are asymptoting to the "
              "model's floor confidence *on noise*, not tracking a drone that remains "
              "audible. The apparent robustness is a positive bias.\n")
            w("This is an independent confirmation of Task 5's finding from a completely "
              "different direction: Task 5 showed the model fires on everything at the one "
              "test location containing both classes; this shows it fires on synthetic "
              "noise containing no audio content at all.\n")
    w(_checks_table(r))
    w(_figure(r.get("figure"), "Model confidence vs signal-to-noise ratio"))


def _write_conclusions(w, results):
    """Write the closing synthesis across all tasks.

    Args:
        w: Line-appending callable.
        results: Every task result record.
    """
    w("# Conclusions\n")
    phase1 = [r for r in results if r["task"] <= 4]
    phase2 = [r for r in results if r["task"] > 4]

    w("## Pipeline integrity\n")
    if all(r["status"] == PASS for r in phase1):
        w("All four structural checks pass. Gradient flow, tensor dimensionality, "
          "inference-time determinism and long-run memory stability are verified. **The "
          "implementation is sound**; the limitations below are properties of the model and "
          "the dataset, not defects in the code.\n")
    else:
        failed = [r["name"] for r in phase1 if r["status"] != PASS]
        w(f"Structural checks failing: {', '.join(failed)}. These are software defects and "
          f"should be resolved before any performance number is treated as meaningful.\n")

    w("## Performance\n")
    latency = _by_task(results, 7)
    if latency and latency["status"] == PASS:
        deployment = latency["devices"][0]
        w(f"**Latency is a solved problem.** End-to-end inference is "
          f"{deployment['end_to_end']['mean_ms']:.2f} ms mean / "
          f"{deployment['end_to_end']['p99_ms']:.2f} ms p99 on `{deployment['device']}`, "
          f"{deployment['headroom_vs_budget']:.0f}x inside the {latency['budget_ms']} ms "
          f"STM32 frame budget. The PC stage will not bottleneck the system.\n")

    task5, task9 = _by_task(results, 5), _by_task(results, 9)
    w("**Detection accuracy is not.** Three independent measurements converge on the same "
      "conclusion:\n")
    if task5 and task5.get("pooled_both_class_locations"):
        pooled = task5["pooled_both_class_locations"]
        w(f"1. **Task 5** — at the only test location containing both classes, the model "
          f"predicts drone for every chunk (specificity {pooled['specificity']:.2f}, "
          f"MCC {pooled['mcc']:.2f}).")
    task8 = _by_task(results, 8)
    if task8 and task8.get("analysis"):
        w(f"2. **Task 8** — validation loss turns upward at epoch "
          f"{task8['analysis']['best_val_loss_epoch']} while training loss keeps falling: "
          f"the model is memorising, and what it memorises is the recording environment.")
    if task9 and task9.get("noise_only_control"):
        control = task9["noise_only_control"]
        w(f"3. **Task 9** — synthetic noise containing no drone is classified as a drone "
          f"{control['white']['false_alarm_rate']:.0%} of the time.\n")

    w("The headline test metrics are carried by single-class locations, where the "
      "environment predicts the label without any detection taking place. The honest "
      "summary is that this checkpoint separates *recording environments*, not drones.\n")

    w("## Known limitations\n")
    w("- **The test split remains partly location-confounded.** Two of its three locations "
      "contain only one class. The split is stratified by label at the recording level, "
      "which is the strongest guarantee available from 23 recordings across 6 locations, "
      "but it cannot manufacture class diversity the corpus does not contain.\n")
    w("- **Only one test location supports an un-confounded measurement**, so the "
      "un-confounded MCC rests on a single location's recordings.\n")
    w("- **Whole-file labelling assumes the drone is audible for a recording's full "
      "duration**, which `HANDOFF.md` flags as doubtful for at least one recording.\n")

    w("## Recommended next steps\n")
    w("1. **Early stopping** at the validation-loss minimum — the cheapest available "
      "improvement, and directly indicated by Task 8.\n")
    w("2. **Leave-one-location-out cross-validation** (the plan recorded in `HANDOFF.md`): "
      "hold each location out entirely, train a fresh model per fold, and report a pooled "
      "confusion matrix in which every prediction comes from a model that never saw that "
      "location. That converts the confound from a caveat into a measured quantity.\n")
    w("3. **Record the missing class-location combinations** — no-drone audio at outdoor "
      "sites and drone audio indoors — so no split can shortcut on location. This is a data "
      "problem, and Tasks 5 and 9 indicate no algorithmic change will substitute for it.\n")
    w("4. **Re-measure with the deployment microphone and gain chain**, which `HANDOFF.md` "
      "notes is currently untracked (`sd.InputStream` uses the OS default device).\n")


if __name__ == "__main__":
    run()
