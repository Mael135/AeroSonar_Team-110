"""Live drone detection from a microphone input stream.

Captures audio into a rolling one-second window, converts each window to a mel
spectrogram, classifies it with the trained CNN, and prints the running detection
state. Requires the optional ``sounddevice`` dependency and an available input device.

Run from the repository root::

    python -m aerosonar.inference.inference
"""
import os
import queue
import torch
import numpy as np
import sounddevice as sd
import yaml
from pathlib import Path
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.inference.streaming import push_frame
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.config import load_default_config


def load_threshold(weights_dir: Path, default: float = 0.5) -> float:
    """Read the tuned detection threshold saved alongside the model weights.

    Args:
        weights_dir: Directory expected to contain ``threshold.yaml``.
        default: Value returned when the file is absent or lacks the key.

    Returns:
        float: The detection threshold to apply to the drone probability.
    """
    thresh_path = weights_dir / "threshold.yaml"
    if thresh_path.exists():
        with open(thresh_path) as f:
            return float(yaml.safe_load(f).get("detection_threshold", default))
    return default


def run_inference():
    """Run the live detection loop until interrupted.

    Opens an input stream on the default audio device, maintains a rolling window of
    the most recent second of audio, and continuously reports the drone probability
    and the resulting detection decision. Blocks until the process is interrupted.

    The audio device is not specified explicitly, so capture uses the operating-system
    default. Detection behaviour depends on the microphone and gain chain in use.
    """
    config = load_default_config()

    SR          = config["data"]["sample_rate"]
    DURATION    = config["data"]["duration"]
    WINDOW_SIZE = int(SR * DURATION)
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir  = Path(config["paths"]["weights"])
    weights_path = weights_dir / "CNN_best.pth"
    threshold    = load_threshold(weights_dir)
    print(f"Using detection threshold: {threshold:.2f}")

    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(DEVICE)
    transform.amplitude_to_db = transform.amplitude_to_db.to(DEVICE)

    model = SpectrogramCNN().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval()

    audio_q = queue.Queue()
    buffer  = np.zeros(WINDOW_SIZE)
    samples_received = 0

    def audio_callback(indata, frames, time_info, status):
        """Hand captured samples to the main loop. Runs on the audio thread."""
        if status:
            print(status)
        audio_q.put(indata.copy())

    print("--- Starting Live Detection (Ctrl+C to stop) ---")
    with sd.InputStream(channels=1, samplerate=SR, callback=audio_callback):
        while True:
            while not audio_q.empty():
                data = audio_q.get()
                push_frame(buffer, data)
                samples_received += len(data)

            # Skip inference until the buffer has filled at least once
            if samples_received < WINDOW_SIZE:
                continue

            audio_tensor = torch.from_numpy(buffer).float().to(DEVICE)
            with torch.no_grad():
                spec   = transform(audio_tensor.unsqueeze(0)).unsqueeze(0)
                logits = model(spec)
                probs  = torch.softmax(logits, dim=1)
                drone_prob = probs[0][1].item()
                detected   = drone_prob > threshold

            label = "DRONE DETECTED!" if detected else "Searching..."
            print(f"\r[{label}] Drone probability: {drone_prob:.2f}", end="", flush=True)


if __name__ == "__main__":
    run_inference()
