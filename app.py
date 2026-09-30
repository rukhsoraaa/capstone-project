import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = "artifacts/best_model.pt"
MAX_FRAMES = 600
MOVEMENT_THRESHOLD_PX = 5
WAITING_THRESHOLD_SEC = 2.0
VEHICLE_KEYWORDS = ["car", "truck", "bus", "motorbike", "motorcycle", "van", "vehicle", "bike"]

st.set_page_config(page_title="Smart Traffic Light", page_icon="🚦", layout="wide")

st.title("🚦 Smart Traffic Light")
st.caption("Adaptive Signal Control via Vehicle Detection")
st.markdown(
    "Upload a traffic video and the trained YOLOv8 + ByteTrack pipeline will "
    "estimate waiting vehicles and recommend a green-light duration."
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

def validate_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError("File not found: " + video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open video (corrupted or unsupported format).")
    ret, _ = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Video contains no readable frames.")
    return True

def vehicle_class_ids(model):
    ids = [i for i, name in model.names.items()
           if any(k in str(name).lower() for k in VEHICLE_KEYWORDS)]
    return ids if ids else None

def process_traffic_video(video_path, model, output_path, max_frames=MAX_FRAMES):
    class_ids = vehicle_class_ids(model)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Could not read video dimensions.")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    stationary_frames, prev_positions, history = {}, {}, []
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=class_ids, verbose=False)
        waiting_count = 0
        total_count = 0

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            total_count = len(track_ids)

            for box, tid in zip(boxes, track_ids):
                cx, cy = float(box[0]), float(box[1])
                if tid in prev_positions:
                    px, py = prev_positions[tid]
                    if np.hypot(cx - px, cy - py) < MOVEMENT_THRESHOLD_PX:
                        stationary_frames[tid] = stationary_frames.get(tid, 0) + 1
                    else:
                        stationary_frames[tid] = 0
                prev_positions[tid] = (cx, cy)
                if stationary_frames.get(tid, 0) / fps > WAITING_THRESHOLD_SEC:
                    waiting_count += 1

        annotated = results[0].plot()
        cv2.putText(annotated, "Vehicles detected: " + str(total_count),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated, "Waiting in queue: " + str(waiting_count),
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        out.write(annotated)
        history.append({"frame": frame_idx, "total": total_count, "waiting": waiting_count})
        frame_idx += 1

    cap.release()
    out.release()
    if frame_idx == 0:
        raise ValueError("No readable frames were processed.")
    return pd.DataFrame(history), fps, frame_idx

def decide_signal(waiting_count, base_green=30, extend_step=5, max_green=60, min_green=15):
    if waiting_count >= 8:
        return min(base_green + extend_step * 2, max_green), "Heavy queue - extend green"
    if waiting_count >= 4:
        return min(base_green + extend_step, max_green), "Moderate queue - extend green slightly"
    if waiting_count <= 1:
        return max(base_green - extend_step, min_green), "Almost empty - shorten green"
    return base_green, "Normal load - keep default timing"

with st.sidebar:
    st.header("Project")
    st.write("YOLOv8 + ByteTrack")
    st.write("Queue estimation + rule-based signal control")
    st.divider()
    st.metric("mAP@50", "0.8820")
    st.metric("Precision", "0.8826")
    st.metric("Recall", "0.8232")

uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])
st.info("Demo limit: the app processes up to 600 frames to keep cloud runtime manageable.")

if uploaded is not None:
    if not os.path.exists(MODEL_PATH):
        st.error("The trained model is missing. Expected: artifacts/best_model.pt")
        st.stop()

    if st.button("🚀 Analyze Traffic", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, uploaded.name)
            output_path = os.path.join(tmpdir, "annotated_output.mp4")
            with open(input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            try:
                validate_video(input_path)
                with st.spinner("Loading trained YOLOv8 model..."):
                    model = load_model()
                with st.spinner("Running YOLOv8 + ByteTrack analysis..."):
                    history, fps, processed_frames = process_traffic_video(
                        input_path, model, output_path
                    )

                window = max(1, int(fps * 5))
                demo_waiting = history["waiting"].tail(window).mean()
                green_time, decision = decide_signal(round(demo_waiting))

                st.success("Analysis complete.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frames processed", processed_frames)
                c2.metric("Vehicles in final frame", int(history["total"].iloc[-1]))
                c3.metric("Waiting vehicles", round(demo_waiting))
                c4.metric("Green duration", f"{green_time}s")

                st.subheader("🚦 Signal Recommendation")
                if "Heavy" in decision:
                    st.error(f"**{decision}** — recommended green: **{green_time} seconds**")
                elif "Moderate" in decision:
                    st.warning(f"**{decision}** — recommended green: **{green_time} seconds**")
                elif "Almost empty" in decision:
                    st.info(f"**{decision}** — recommended green: **{green_time} seconds**")
                else:
                    st.success(f"**{decision}** — recommended green: **{green_time} seconds**")

                st.subheader("🎥 Annotated Traffic Video")
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button(
                    "⬇️ Download annotated video",
                    data=video_bytes,
                    file_name="smart_traffic_light_output.mp4",
                    mime="video/mp4",
                )

                st.subheader("📊 Waiting Vehicles Over Time")
                st.line_chart(history.set_index("frame")[["waiting"]])

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.caption("Check the video and make sure artifacts/best_model.pt is present.")

else:
    st.subheader("How it works")
    a, b, c, d = st.columns(4)
    a.markdown("**1. Video**\n\nTraffic camera footage")
    b.markdown("**2. YOLOv8**\n\nVehicle detection")
    c.markdown("**3. ByteTrack**\n\nVehicle tracking + waiting")
    d.markdown("**4. Signal logic**\n\nGreen-time recommendation")

    st.subheader("Reported model results")
    r1, r2, r3 = st.columns(3)
    r1.metric("mAP@50", "88.20%")
    r2.metric("Precision", "88.26%")
    r3.metric("Recall", "82.32%")
    st.caption("Prototype only: this recommendation must not directly control a real traffic signal.")

