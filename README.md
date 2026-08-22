# 💡 AeroSonar

<!-- cool project cover image -->
![Project Cover Image](/media/AeroSonar_cover.jpeg)

<!-- table of content -->
## Table of Contents
- [The Team](#-the-team)
- [Project Description](#-project-description)
- [Repository Layout](#-repository-layout)
- [Getting Started](#-getting-started)
- [Prerequisites](#-prerequisites)
- [Installing](#-installing)
- [Testing](#-testing)
- [Results](#-results)
- [Deployment](#-deployment)
- [Known Limitations](#-known-limitations)
- [Built With](#-built-with)
- [Acknowledgments](#-acknowledgments)

## 👥 The Team

**Team Members**
- [Matan Elad](mailto:matan.elad@mail.huji.ac.il)
- [Guy Regev](mailto:regev.guy@mail.huji.ac.il)

**Supervisor**
- [Gal Katzhendler](mailto:gal.katzhendler@mail.huji.ac.il)

## 📚 Project Description

Small UAVs evade radar and radio-frequency detection through their low radar
cross-section and, under autonomous flight, the absence of RF emissions.
AeroSonar is a standalone acoustic unit that detects a drone and reports the
direction it is approaching from. A single microphone feeds a convolutional
neural network that verifies the acoustic presence of a drone, while eight
microphones on a 9.5 cm circle feed an STM32H753 microcontroller that
estimates bearing by SRP-PHAT across all 28 microphone pairs. Because the two
paths share no processor, spatial tracking adds no load to the classifier.

**Features**
- Bearing estimation at 24 updates per second, entirely on-device
- Drone classification with majority voting across inference windows
- Timer-paced acquisition with DMA transfer, no processor involvement
- Steering geometry self-verified at initialisation, without acoustic input
- Under 20% utilisation of a single Cortex-M7 core

**Components**
- Eight-element uniform circular microphone array, 9.5 cm radius
- Single reference microphone for classification
- STM32H753ZI on a NUCLEO-144 board
- Host machine for network training and inference

**Technologies**
- SRP-PHAT with GCC-PHAT cross-correlation for direction of arrival
- CMSIS-DSP real FFT on Cortex-M7 with hardware floating point
- STM32 HAL, timer-triggered ADC, BDMA circular transfer, MPU cache control
- PyTorch

## ⚡ Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes.

### 🧱 Prerequisites

- [STM32CubeIDE 2.2.0](https://www.st.com/en/development-tools/stm32cubeide.html)
- [STM32Cube MCU package for STM32H7](https://www.st.com/en/embedded-software/stm32cubeh7.html)
- [NUCLEO-H753ZI](https://www.st.com/en/evaluation-tools/nucleo-h753zi.html)
- [Python 3.10+](https://www.python.org/) and [PyTorch](https://pytorch.org/)
- 8x [PUI AOM-5024L-HD-R](https://puiaudio.com/) capsules and eight 2.2 kΩ
  resistors
- 1x digital USB microphone
- 1x host PC

### 🏗️ Installing

Clone the repository

    git clone https://github.com/<user>/aerosonar.git
    cd aerosonar

Open the firmware in STM32CubeIDE via `File → Open Projects from File System`,
selecting `embedded/`. Build with optimisation enabled; the default debug
configuration runs roughly 2.8× slower.

Wire each capsule through a 2.2 kΩ resistor to 3.3 V, connecting the junction
directly to a converter input. Conversion rank order must match the ring
positions in `MIC_CH[]`

    Rank 1: IN2  (PF9)     Rank 5: IN7  (PF8)
    Rank 2: IN3  (PF7)     Rank 6: IN8  (PF6)
    Rank 3: IN4  (PF5)     Rank 7: IN9  (PF4)
    Rank 4: IN6  (PF10)    Rank 8: IN10 (PC0)

Flash with `Run → Debug As → STM32 C/C++ Application`. On Linux, install the
ST-LINK udev rules first

    sudo cp ~/st/stm32cubeide_*/scripts/*.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger

Set up the classification environment

    python -m venv .venv && source .venv/bin/activate
    pip install -e .

With the firmware running, add `doa_res.azimuth_deg` and `doa_res.confidence`
to Live Expressions. Broadband noise near the array produces a non-zero
confidence ratio and a bearing that follows the source.

***Detection pipeline***

**1. Preprocess.** Segments the raw recordings in `data/final_raw/` into one-second
chunks, converts each to a loudness-normalised decibel mel spectrogram, and writes them
to `data/processed/` along with `metadata.csv` and `expanded_metadata.csv`. Re-run this
whenever the `data` or `spectrogram` settings in `default.yaml` change.

```bash
python -m aerosonar.data.preprocessData
```

**2. Train.** Trains the CNN on a label-stratified, recording-level train/validation/test
split. Validation drives checkpoint selection and threshold tuning; the test split is
never read during training. Writes `CNN_best.pth` and `threshold.yaml` to
`src/aerosonar/models/weights/`, the per-epoch history to `reports/eval/train_history.csv`,
and figures to `graphs/`.

```bash
python -m aerosonar.training.trainCNN
```

**3. Inference.**
For live detection from a microphone, run `python -m aerosonar.inference.inference`.

## 🧪 Testing

Tests run on the target and report through instrumentation variables, read in
the debugger's Expressions view. Each has an outcome predicted independently
of the implementation.

### Sample Tests

**Localization.** `doa_smoke_test()` transforms a sinusoid of exactly eight
cycles across the window, placing all energy in one bin. It confirms the
floating-point unit is active, the transform tables are linked, and the
headers match the compiled objects — none detectable at build time.

    g_peak_idx == 8

`DOA_Init()` checks the steering table against closed-form values derived from
the array geometry, requiring no acoustic input.

    dbg_max_lag      13.85 samples   array diameter / speed of sound
    dbg_pair0_at0    -2.03 samples   adjacent pair, source at 0°
    dbg_diam_at0     -13.85 samples  antipodal pair, source at 0°
    dbg_geo_antisym  ~1e-6           geometric antisymmetry residual

`ch_dc[]` and `ch_pp[]` confirm every channel is biased and responsive;
`doa_overruns` must stay at zero, and `doa_cycles` reports the per-frame cost.

**Detection.** Evaluation on the held-out set reports the operating point
selected by a post-hoc threshold sweep.

    python -m aerosonar.evaluate --checkpoint CNN_best.pth
    
    threshold 0.90   precision 0.969   recall 0.777   F1 0.862

Per-frame and per-event recall are reported separately. Majority voting across
a rolling window converts independent per-frame errors into far fewer missed
events, and the gap between the two quantifies what the smoothing contributes.

A parity check asserts that the inference script's feature extraction matches
the training pipeline for identical input, guarding against silent divergence
in windowing or normalisation.

## 🚀 Deployment

The unit runs standalone: acquisition and processing begin at power-on with no
host connection. Bearing and confidence are available in `doa_res` and can be
transmitted over the on-board virtual COM port. Localization runs on dedicated
hardware, so the two subsystems can be deployed on separate processors
unchanged. Detection range is currently limited by the analog front end, which
has no preamplifier.

## ⚙️ Built With

  - [CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) - Real FFT and
    complex arithmetic on Cortex-M7
  - [STM32Cube HAL](https://www.st.com/en/embedded-software/stm32cubeh7.html) -
    Peripheral abstraction for the STM32H7
  - [PyTorch](https://pytorch.org/) - Network training and inference

## 🙏 Acknowledgments

  - Knapp and Carter (1976), for the phase transform
  - DiBiase (2000), for the steered-response power formulation
  - The CMSIS-DSP maintainers
  - Our supervisor, Gal, for guidance on the ideas and concepts which we used.

