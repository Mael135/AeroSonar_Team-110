# Handoff: real-world drone detection failure investigation

## Symptom

CNN trained on STFT/Mel spectrograms from studio-mic recordings performed great in
training/validation but **extremely poorly in real-world testing**.

## Root causes found, in order of impact

1. **No loudness normalization (fixed).** `SpectrogramTransform` computed dB-Mel
   spectrograms directly off raw waveform amplitude, with no level normalization. The
   only invariance came from a ±6 dB augmentation in `TrainAugment` — far too narrow to
   cover real differences in mic gain knob position, OS input volume, AGC, or distance.
   Evidence: the tuned detection threshold was sitting at 0.10, a sign of poor
   calibration.

2. **Location/label confound (partially mitigated, not solved).** The raw dataset
   (`data/edited_raw`, 22-23 recordings) has several single-class locations: `room` and
   `balcony` are 100% no-drone, `ben_shemen` is 100% drone. Worse, the random
   recording-level train/test split (seed=42) happens to put an entire single-class
   location in every test fold — confirmed via `src/aerosonar/utils/audit.py`, which
   showed the test set's "near-perfect" F1 was actually measuring "can the model tell
   `room` apart from outdoor locations," not real drone detection.

3. **Possible label noise (flagged, not yet cleaned).** Whole-file labeling assumes a
   "drone" recording has the drone audible for its entire duration. The audit's
   prominence heuristic flagged 79 drone-labeled chunks as weak candidates for an
   audible drone — 54 of them from a single recording (`file_id 21`). Worth
   spot-listening to that file.

## What's been implemented

- **`src/aerosonar/features/transforms.py`**: `SpectrogramTransform._normalize_loudness()`
  — RMS-normalizes each waveform chunk to a fixed target level (`-20 dBFS`, capped at
  `+30 dB` max gain) before the mel transform. Config knobs in `default.yaml`
  (`normalize_target_dbfs`, `normalize_max_gain_db`). Verified: a 34 dB gain difference
  now produces numerically identical spectrograms.
- **`src/aerosonar/data/dataset.py`**: `CrossSessionMixer` — during training, additively
  mixes (in spectrogram power domain) a no-drone background from a *different*
  recording/location into each training sample, regardless of its own label. This
  synthesizes "drone heard over a background it was never actually recorded against"
  (e.g. drone-over-room-background), directly attacking the location confound without
  needing new recordings. Background pool is drawn from the train split only — no test
  leakage. `build_dataloaders()` now joins `expanded_metadata.csv` for location and
  wires the mixer through `ApplyTransform` → `TrainAugment`.
- **`src/aerosonar/utils/audit.py`**: standalone diagnostic sweep (`python -m
  aerosonar.utils.audit`) over raw audio (clipping/duration/sr checks), dataset
  composition (location×label crosstabs, single-class location detection), spectrogram
  level stats vs. recording gain, chunk-level drone-presence heuristics, and the trained
  model's per-chunk/per-location prediction breakdown. Writes CSVs to `reports/audit/`.
  **This is the tool that found root causes #2 and #3 — rerun it after any retrain.**

## Current state (as of this checkpoint)

- Reprocessed `data/processed/*.pt` with the normalization fix, retrained twice.
- Latest tuned threshold: **0.13** (up from 0.06 pre-mixing, 0.10 pre-normalization —
  moving in the right direction, less pathologically low).
- Per-location test breakdown still shows every location as single-class within the
  test fold (unchanged by the mixing fix, since mixing only touches training data, not
  the test split's composition). One genuine within-location error appeared
  (`ben_shemen`, 1 false negative) that wasn't there before mixing — a weak but real
  signal that the model isn't purely pattern-matching location anymore.
- **Caveat: the test split itself is still fully location-confounded, so none of these
  numbers can yet be trusted as a real generalization estimate.**

## Next step (not yet implemented)

**Leave-one-location-out cross-validation.** Plan agreed on but not coded:
1. Refactor `build_dataloaders()` into a lower-level function taking explicit
   `train_file_ids`/`test_file_ids`, with the current seed-42 random split becoming a
   thin wrapper (keeps `trainCNN.py` working unchanged).
2. New `src/aerosonar/training/cross_validate.py`: loop over the 6 locations, hold each
   out entirely (no training exposure, not even as a mixing background), train a fresh
   `SpectrogramCNN` per fold, evaluate on the held-out location at threshold=0.5 (no
   per-fold threshold tuning — that would reintroduce the calibration-gaming problem).
3. Report per-fold metrics (note: `room`/`balcony` folds only have a false-positive-rate
   defined, `ben_shemen` only has recall defined, since they're single-class) plus a
   **pooled confusion matrix across all folds** as the headline trustworthy number —
   every prediction in it comes from a model that never saw that location at all.

## Longer-term (needs new data, not algorithmic)

- Record no-drone audio in `ben_shemen`-like outdoor spots, and drone audio in
  `room`/`balcony`-like indoor spots — every location should eventually have both
  classes so no split can shortcut on location.
- Pin down and log the exact mic/input device used in `inference.py` / `app.py`
  (`sd.InputStream` currently relies on the OS default device with no explicit
  `device=`) — re-test with the *same physical mic/gain chain* as training before
  trusting any new checkpoint's real-world behavior.
