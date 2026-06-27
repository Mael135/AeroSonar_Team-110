# import os
# import math
# import numpy as np
# import streamlit as st
# import pydeck as pdk

# # -------------------------
# # Geometry helpers
# # -------------------------

# EARTH_RADIUS_M = 6371000.0

# def wrap_deg(x: float) -> float:
#     x = x % 360.0
#     return x + 360.0 if x < 0 else x

# def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float):
#     """
#     Great-circle destination point given start lat/lon, bearing, distance.
#     Returns (lat, lon) in degrees.
#     """
#     lat1 = math.radians(lat_deg)
#     lon1 = math.radians(lon_deg)
#     brng = math.radians(bearing_deg)
#     dr = distance_m / EARTH_RADIUS_M

#     lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(brng))
#     lon2 = lon1 + math.atan2(
#         math.sin(brng) * math.sin(dr) * math.cos(lat1),
#         math.cos(dr) - math.sin(lat1) * math.sin(lat2)
#     )

#     return math.degrees(lat2), math.degrees(lon2)

# def circle_polygon(lat: float, lon: float, radius_m: float, n: int = 120):
#     """Approximate a circle as a polygon (list of [lon, lat])."""
#     pts = []
#     for b in np.linspace(0, 360, n, endpoint=False):
#         lat2, lon2 = destination_point(lat, lon, float(b), radius_m)
#         pts.append([lon2, lat2])
#     pts.append(pts[0])  # close ring
#     return pts

# def wedge_polygon(lat: float, lon: float, center_bearing_deg: float, half_angle_deg: float, radius_m: float, n: int = 40):
#     """
#     Sector / wedge polygon:
#     - Start at device point
#     - Go along arc from (center-half_angle) to (center+half_angle)
#     - Back to device point
#     Output: list of [lon, lat]
#     """
#     start = center_bearing_deg - half_angle_deg
#     end = center_bearing_deg + half_angle_deg

#     arc = []
#     for b in np.linspace(start, end, n):
#         lat2, lon2 = destination_point(lat, lon, float(b), radius_m)
#         arc.append([lon2, lat2])

#     return [[lon, lat]] + arc + [[lon, lat]]  # close at device


# # -------------------------
# # Streamlit UI
# # -------------------------

# st.set_page_config(page_title="Drone Direction Map MVP", layout="wide")

# token = os.getenv("MAPBOX_TOKEN")
# if not token:
#     st.error("MAPBOX_TOKEN environment variable is not set. Set it and rerun.")
#     st.stop()

# st.title("Drone Detection Direction on Map (Mapbox + Python)")

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Device location")
#     lat = st.number_input("Latitude", value=31.7780, format="%.6f")   # Jerusalem-ish default
#     lon = st.number_input("Longitude", value=35.2350, format="%.6f")

# with col2:
#     st.subheader("Device heading / azimuth")
#     device_azimuth = st.slider("Device azimuth (deg, 0=N)", 0.0, 360.0, 0.0, 0.1)

# with col3:
#     st.subheader("Detection")
#     bearing_mode = st.selectbox("Bearing mode", ["Relative to device forward", "Absolute (true north)"])
#     detected_bearing = st.slider("Detected bearing (deg)", 0.0, 360.0, 45.0, 0.1)
#     confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.01)

# # Compute absolute bearing
# if bearing_mode == "Relative to device forward":
#     bearing_abs = wrap_deg(device_azimuth + detected_bearing)
# else:
#     bearing_abs = wrap_deg(detected_bearing)

# # Overlay configuration
# st.sidebar.header("Overlay settings")
# ring_radius_m = st.sidebar.slider("Ring radius (meters)", 50, 3000, 800, 50)
# wedge_radius_m = st.sidebar.slider("Wedge radius (meters)", 50, 5000, 1200, 50)
# half_angle = st.sidebar.slider("Wedge half-angle (deg)", 1.0, 30.0, 8.0, 0.5)
# auto_zoom = st.sidebar.checkbox("Auto zoom to device", value=True)
# pitch = st.sidebar.slider("Map pitch", 0, 70, 45, 1)

# # Create geometries
# ring = circle_polygon(lat, lon, ring_radius_m)
# wedge = wedge_polygon(lat, lon, bearing_abs, half_angle, wedge_radius_m)

# # Data for layers
# device_point = [{"position": [lon, lat], "label": "Device"}]
# ring_poly = [{"polygon": ring}]
# wedge_poly = [{
#     "polygon": wedge,
#     "confidence": confidence,
#     "bearing_abs": bearing_abs
# }]

# # Color logic (simple): higher confidence => more opaque
# # pydeck wants RGBA [0..255]; we keep it simple and readable.
# alpha = int(60 + 160 * confidence)  # 60..220
# wedge_fill = [255, 0, 0, alpha]     # red with alpha

# layers = [
#     # Wedge sector
#     pdk.Layer(
#         "PolygonLayer",
#         data=wedge_poly,
#         get_polygon="polygon",
#         get_fill_color=wedge_fill,
#         get_line_color=[255, 0, 0, 255],
#         line_width_min_pixels=2,
#         pickable=True,
#     ),
#     # 360 ring
#     pdk.Layer(
#         "PolygonLayer",
#         data=ring_poly,
#         get_polygon="polygon",
#         get_fill_color=[0, 0, 0, 0],
#         get_line_color=[0, 255, 255, 180],
#         line_width_min_pixels=2,
#         pickable=False,
#     ),
#     # Device marker
#     pdk.Layer(
#         "ScatterplotLayer",
#         data=device_point,
#         get_position="position",
#         get_radius=10,
#         radius_min_pixels=6,
#         get_fill_color=[0, 255, 0, 220],
#         pickable=True,
#     ),
# ]

# tooltip = {
#     "html": "<b>Bearing:</b> {bearing_abs}°<br/><b>Confidence:</b> {confidence}",
#     "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
# }

# view_state = pdk.ViewState(
#     latitude=lat,
#     longitude=lon,
#     zoom=16 if auto_zoom else 12,
#     pitch=pitch,
#     bearing=0,  # keep map north-up; overlays show direction
# )

# deck = pdk.Deck(
#     map_style="mapbox://styles/mapbox/satellite-streets-v12",
#     initial_view_state=view_state,
#     layers=layers,
#     tooltip=tooltip,
# )

# st.pydeck_chart(deck, width=True)

# st.markdown(
#     f"""
# **Computed absolute bearing:** `{bearing_abs:.1f}°`  
# (0° = North, clockwise positive)
# """
# )



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

# --- 1. Shared State Bridge ---
class DetectionState:
    def __init__(self):
        self.is_drone = False
        self.confidence = 0.0
        self.last_update = time.time()
        self.audio_source = "microphone"
        self.error = None

@st.cache_resource
def get_shared_state():
    return DetectionState()

# --- 2. Geometry Helper for the Cone ---
def get_wedge_polygon(lat, lon, azimuth, radius=1000, angle=30):
    """Generates a wedge/cone polygon starting from the device."""
    coords = [[lon, lat]]
    steps = 20
    start_angle = azimuth - (angle / 2)
    end_angle = azimuth + (angle / 2)
    
    for i in range(steps + 1):
        curr_angle = start_angle + (i * (end_angle - start_angle) / steps)
        rad = math.radians(curr_angle)
        # Simple lat/lon approximation for the radius
        dx = radius * math.sin(rad) / 111320.0
        dy = radius * math.cos(rad) / 110540.0
        coords.append([lon + dx, lat + dy])
    
    coords.append([lon, lat])
    return coords

def has_input_device() -> bool:
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            return True
        return any(d["max_input_channels"] > 0 for d in sd.query_devices())
    except sd.PortAudioError:
        return False

def load_inference_model(config, device):
    transform = SpectrogramTransform(config)
    transform.mel_spectrogram = transform.mel_spectrogram.to(device)
    transform.amplitude_to_db = transform.amplitude_to_db.to(device)

    model = SpectrogramCNN().to(device)
    weights_path = os.path.join(config["paths"]["weights"], "CNN_best.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return transform, model

def run_inference_step(state, transform, model, device, buffer):
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
    while not audio_q.empty():
        data = audio_q.get()
        n = len(data)
        buffer[:] = np.roll(buffer, -n)
        buffer[-n:] = data.flatten()

def start_microphone_thread(state, config):
    sr = config["data"]["sample_rate"]
    window_size = int(sr * config["data"]["duration"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform, model = load_inference_model(config, device)

    audio_q = queue.Queue()
    buffer = np.zeros(window_size)

    def audio_callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    state.audio_source = "microphone"
    with sd.InputStream(channels=1, samplerate=sr, callback=audio_callback):
        while True:
            drain_audio_queue(buffer, audio_q)
            run_inference_step(state, transform, model, device, buffer)
            time.sleep(0.05)

def start_file_thread(state, config, wav_path: str):
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
    try:
        if has_input_device():
            start_microphone_thread(state, config)
        else:
            start_file_thread(state, config, demo_wav)
    except Exception as exc:
        state.error = str(exc)
        raise

# --- 4. Streamlit UI Layout ---
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