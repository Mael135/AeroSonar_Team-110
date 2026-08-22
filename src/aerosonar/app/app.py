"""Streamlit dashboard for live drone detection.

Runs inference on a background thread while the interface renders the current detection
state over a map. Audio comes from the microphone where one is available, falling back to
a looped WAV file so the dashboard remains demonstrable on hosts without an input device.

Requires the optional ``live`` dependencies (``streamlit``, ``sounddevice``, ``pydeck``).
A ``MAPBOX_TOKEN`` environment variable is needed for the map basemap, and
``AEROSONAR_DEMO_WAV`` overrides the fallback audio file.

Note:
    :func:`run_inference_step` decides using ``argmax``, an implicit threshold of 0.5,
    and does not read the tuned value from ``threshold.yaml`` that
    :mod:`aerosonar.inference.inference` applies. The two runtimes will therefore
    disagree on any sample scoring between 0.5 and the tuned threshold.

Run from the repository root::

    streamlit run src/aerosonar/app/app.py
"""
import os
import time
import math
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import streamlit as st
import pydeck as pdk

from aerosonar.features.transforms import SpectrogramTransform
from aerosonar.models.spectrogramCNN import SpectrogramCNN
from aerosonar.config import load_default_config

DEFAULT_DEMO_WAV = (
    "data/edited_raw/2026-01-19__living-room__drone__low-noise_ambience__g75__22k05__5m__01.wav"
)

# --- Shared state bridge ---
class DetectionState:
    """Detection state shared between the inference thread and the interface.

    Attributes:
        is_drone: Whether the most recent window was classified as a drone.
        confidence: Drone probability for that window.
        last_update: Timestamp of the most recent inference.
        audio_source: Either ``"microphone"`` or ``"file"``.
        error: Message describing why audio capture failed, if it did.
    """

    def __init__(self):
        """Initialise to the no-detection state."""
        self.is_drone = False
        self.confidence = 0.0
        self.last_update = time.time()
        self.audio_source = "microphone"
        self.error = None

@st.cache_resource
def get_shared_state():
    """Return the process-wide detection state, created once per session.

    Returns:
        DetectionState: The shared instance.
    """
    return DetectionState()

def get_wedge_polygon(lat, lon, azimuth, radius=1000, angle=30):
    """Build a wedge polygon fanning out from a point along a bearing.

    Uses a flat-earth approximation, which is accurate enough at the radii involved.

    Args:
        lat: Origin latitude in degrees.
        lon: Origin longitude in degrees.
        azimuth: Centre bearing in degrees, clockwise from north.
        radius: Wedge radius in metres.
        angle: Total angular width in degrees.

    Returns:
        list: Closed ring of ``[longitude, latitude]`` pairs.
    """
    coords = [[lon, lat]]
    steps = 20
    start_angle = azimuth - (angle / 2)
    end_angle = azimuth + (angle / 2)
    
    for i in range(steps + 1):
        curr_angle = start_angle + (i * (end_angle - start_angle) / steps)
        rad = math.radians(curr_angle)
        dx = radius * math.sin(rad) / 111320.0
        dy = radius * math.cos(rad) / 110540.0
        coords.append([lon + dx, lat + dy])
    
    coords.append([lon, lat])
    return coords

def has_input_device() -> bool:
    """Report whether any audio input device is available.

    Returns:
        bool: True if a default or enumerable input device exists.
    """
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            return True
        return any(d["max_input_channels"] > 0 for d in sd.query_devices())
    except sd.PortAudioError:
        return False

def load_inference_model(config, device):
    """Load the feature extractor and trained model onto a device.

    Args:
        config: Project configuration.
        device: Compute device.

    Returns:
        tuple: ``(transform, model)`` with the model in evaluation mode.
    """
    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(device)
    transform.amplitude_to_db = transform.amplitude_to_db.to(device)

    model = SpectrogramCNN().to(device)
    weights_path = os.path.join(config["paths"]["weights"], "CNN_best.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return transform, model

def run_inference_step(state, transform, model, device, buffer):
    """Classify the current audio window and update the shared state.

    Args:
        state: The shared :class:`DetectionState`, updated in place.
        transform: Feature extractor.
        model: Trained model in evaluation mode.
        device: Compute device.
        buffer: The rolling window of audio samples.
    """
    audio_tensor = torch.from_numpy(buffer).float().to(device)
    with torch.no_grad():
        spec = transform(audio_tensor.unsqueeze(0)).unsqueeze(0)
        logits = model(spec)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        state.is_drone = (pred == 1)
        state.confidence = probs[0][1].item()
        state.last_update = time.time()

def drain_audio_queue(buffer, audio_q):
    """Move all queued audio into the rolling window.

    Args:
        buffer: The rolling window, modified in place.
        audio_q: Queue fed by the capture callback.
    """
    while not audio_q.empty():
        data = audio_q.get()
        n = len(data)
        buffer[:] = np.roll(buffer, -n)
        buffer[-n:] = data.flatten()

def start_microphone_thread(state, config):
    """Capture from the microphone and run inference until the process exits.

    Args:
        state: The shared :class:`DetectionState`, updated in place.
        config: Project configuration.
    """
    sr = config["data"]["sample_rate"]
    window_size = int(sr * config["data"]["duration"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform, model = load_inference_model(config, device)

    audio_q = queue.Queue()
    buffer = np.zeros(window_size)

    def audio_callback(indata, frames, time_info, status):
        """Hand captured samples to the inference loop. Runs on the audio thread."""
        audio_q.put(indata.copy())

    state.audio_source = "microphone"
    with sd.InputStream(channels=1, samplerate=sr, callback=audio_callback):
        while True:
            drain_audio_queue(buffer, audio_q)
            run_inference_step(state, transform, model, device, buffer)
            time.sleep(0.05)

def start_file_thread(state, config, wav_path: str):
    """Loop a WAV file through inference, standing in for live capture.

    Args:
        state: The shared :class:`DetectionState`, updated in place.
        config: Project configuration.
        wav_path: File to loop.
    """
    sr = config["data"]["sample_rate"]
    window_size = int(sr * config["data"]["duration"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform, model = load_inference_model(config, device)

    audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    buffer = np.zeros(window_size)
    offset = 0
    chunk_size = max(1, window_size // 10)
    state.audio_source = f"file: {Path(wav_path).name}"

    while True:
        end = offset + chunk_size
        if end >= len(audio):
            chunk = audio[offset:]
            offset = 0
        else:
            chunk = audio[offset:end]
            offset = end

        n = len(chunk)
        buffer[:] = np.roll(buffer, -n)
        buffer[-n:] = chunk
        run_inference_step(state, transform, model, device, buffer)
        time.sleep(0.05)

def start_inference_thread(state, config, demo_wav: str):
    """Start inference on a daemon thread, choosing microphone or file input.

    Args:
        state: The shared :class:`DetectionState`.
        config: Project configuration.
        demo_wav: Fallback WAV file used when no input device is present.
    """
    try:
        if has_input_device():
            start_microphone_thread(state, config)
        else:
            start_file_thread(state, config, demo_wav)
    except Exception as exc:
        state.error = str(exc)
        raise

# --- Streamlit UI layout ---
st.set_page_config(layout="wide", page_title="Drone Detector Live")
config = load_default_config()
state = get_shared_state()

demo_wav = os.environ.get("AEROSONAR_DEMO_WAV", DEFAULT_DEMO_WAV)

# Start background thread once
if "inference_running" not in st.session_state:
    if not has_input_device():
        st.session_state.using_demo_audio = True
    threading.Thread(
        target=start_inference_thread,
        args=(state, config, demo_wav),
        daemon=True,
    ).start()
    st.session_state.inference_running = True

st.title("Aerial Detector - Live Surveillance")

if st.session_state.get("using_demo_audio"):
    st.warning(
        "No microphone found (common on WSL). Running in demo mode using "
        f"`{demo_wav}`. Set `AEROSONAR_DEMO_WAV` to use another file."
    )
if state.error:
    st.error(f"Inference error: {state.error}")

# Sidebar Controls
st.sidebar.header("Device Configuration")
st.sidebar.caption(f"Audio source: {state.audio_source}")
lat_input = st.sidebar.number_input("Latitude", value=31.7780, format="%.6f")
lon_input = st.sidebar.number_input("Longitude", value=35.2350, format="%.6f")
azimuth_input = st.sidebar.slider("Azimuth (Heading)", 0, 360, 0)

# Map Logic
# Green: [0, 255, 0, 100], Red: [255, 0, 0, 150]
cone_color = [255, 0, 0, 150] if state.is_drone else [0, 255, 0, 100]
wedge_data = [{"polygon": get_wedge_polygon(lat_input, lon_input, azimuth_input)}]

view_state = pdk.ViewState(
    latitude=lat_input,
    longitude=lon_input,
    zoom=14,
    pitch=0,
    bearing=0 # North is up; the cone shows direction
)

cone_layer = pdk.Layer(
    "PolygonLayer",
    data=wedge_data,
    get_polygon="polygon",
    get_fill_color=cone_color,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/satellite-v9",
    initial_view_state=view_state,
    layers=[cone_layer],
    api_keys={"mapbox": os.environ.get("MAPBOX_TOKEN")}
))

# Status Display
if state.is_drone:
    st.error(f"### ⚠️ DRONE DETECTED: TAKE SHELTER! (Conf: {state.confidence:.2f})")
else:
    st.success(f"### ✅ Drone not detected. (Conf: {state.confidence:.2f})")

# Auto-refresh UI
time.sleep(0.1)
st.rerun()