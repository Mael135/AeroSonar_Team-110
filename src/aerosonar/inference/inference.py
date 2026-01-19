import torch
import numpy as np
import sounddevice as sd
import queue
import os
from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.config import load_default_config
config = load_default_config()

SR = config["data"]["sample_rate"]
DURATION = config["data"]["duration"]
WINDOW_SIZE = int(SR * DURATION)
STEP_SIZE = int(SR * 1.0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


audio_q = queue.Queue()
buffer = np.zeros(WINDOW_SIZE)

def audio_callback(indata, frames, time, status):
    """This is called by sounddevice for every new chunk of audio."""
    if status:
        print(status)
    audio_q.put(indata.copy())

spectrogramTransform = SpectrogramTransform(config)
spectrogramTransform.mel_spectrogram = spectrogramTransform.mel_spectrogram.to(DEVICE)
spectrogramTransform.amplitude_to_db = spectrogramTransform.amplitude_to_db.to(DEVICE)

model = SpectrogramCNN().to(DEVICE)
weights_path = os.path.join(config["paths"]["weights"], "CNN_best.pth")
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()

print("--- Starting Live Detection ---")
stream = sd.InputStream(channels=1, samplerate=SR, callback=audio_callback)

with stream:
    while True:
        # Get new data from queue
        while not audio_q.empty():
            data = audio_q.get()
            # Roll buffer: remove oldest data, add newest
            buffer = np.roll(buffer, -len(data))
            buffer[-len(data):] = data.flatten()

        # Convert buffer to tensor
        audio_tensor = torch.from_numpy(buffer).float().to(DEVICE)
        
        # Create Spectrogram (Result shape: [1, Freq, Time])
        with torch.no_grad():
            input_signal = audio_tensor.unsqueeze(0)

            spec = spectrogramTransform(input_signal).unsqueeze(0) # Add Batch dim
            
            # Run Inference
            logits = model(spec)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
            # Print result
            label = "DRONE DETECTED!" if pred == 1 else "Searching..."
            confidence = probs[0][pred].item()
            print(f"\r[{label}] Confidence: {confidence:.2f}", end="")
