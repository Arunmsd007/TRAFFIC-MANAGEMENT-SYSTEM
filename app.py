import streamlit as st
import cv2
import pandas as pd
import time
from collections import deque
import numpy as np
from ultralytics import YOLO
from src.traffic import TrafficController
import os

# ----------------------------
# Vehicle Detector
# ----------------------------
class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)
        lane_counts = {"A":0,"B":0,"C":0,"D":0}
        emergency_detected = False
        annotated_frame = frame.copy()
        frame_width = frame.shape[1]

        for box in results[0].boxes:
            cls = int(box.cls)
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]

            if cls in [2,3,5,7]:
                label = {2:"Car",3:"Bike",5:"Bus",7:"Truck"}[cls]
                x_center = (xyxy[0]+xyxy[2])//2
                if x_center < frame_width*0.25: lane="A"
                elif x_center < frame_width*0.5: lane="B"
                elif x_center < frame_width*0.75: lane="C"
                else: lane="D"

                lane_counts[lane]+=1
                if cls==5: emergency_detected=True
                cv2.rectangle(annotated_frame,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]),(0,255,0),2)
                cv2.putText(annotated_frame,label,(xyxy[0],xyxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        return lane_counts, emergency_detected, annotated_frame

# ----------------------------
# Page Styling
# ----------------------------
st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.markdown("""
<style>
body {background-color:#0f172a; color:white;}
h1,h2,h3,h4,h5,h6 {color:#38bdf8;}
.metric-container {background:#1e293b; padding:12px; border-radius:10px; margin-bottom:8px;}
.signal-box {background:#1e293b; padding:10px; border-radius:12px; display:inline-block; margin:5px;}
.signal-circle {border-radius:50%; width:30px; height:30px; display:inline-block; margin:3px;}
.congestion-low {background-color:green; color:white; padding:2px 8px; border-radius:5px;}
.congestion-medium {background-color:orange; color:white; padding:2px 8px; border-radius:5px;}
.congestion-high {background-color:red; color:white; padding:2px 8px; border-radius:5px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🚦 Smart Traffic Management Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Controls")
use_rl = st.sidebar.checkbox("Use RL Controller", value=False)
video_path = st.sidebar.text_input("Video Path", "data/traffic.mp4")
run_btn = st.sidebar.button("▶ Start Simulation")
override_lane = st.sidebar.selectbox("Manual Override Lane", ["None","A","B","C","D"])
override_phase = st.sidebar.selectbox("Phase", ["green","yellow","red"])
apply_override = st.sidebar.button("Apply Override")

# ----------------------------
# Load Controller
# ----------------------------
if use_rl:
    from stable_baselines3 import DQN
    rl_model = DQN.load("models/rl_dqn")
else:
    controller = TrafficController()

detector = VehicleDetector()

# ----------------------------
# Check video file exists
# ----------------------------
if not os.path.exists(video_path):
    st.error(f"❌ Video file not found: {video_path}")
    st.stop()

cap = cv2.VideoCapture(video_path)
if "history" not in st.session_state: st.session_state.history = deque(maxlen=500)

# ----------------------------
# Layout Columns
# ----------------------------
video_col, info_col = st.columns([3,2])
with video_col:
    st.subheader("📹 Live Video Feed")
    frame_placeholder = st.empty()

with info_col:
    st.subheader("📊 Lane Vehicle Counts")
    metric_cols = st.columns(4)
    metric_objects = {lane: col.metric(f"Lane {lane}", "—") for lane,col in zip(["A","B","C","D"], metric_cols)}
    st.subheader("📈 Traffic Density Trend")
    chart_placeholder = st.empty()

st.markdown("### 📝 Recent Event Logs")
log_placeholder = st.empty()

# ----------------------------
# Traffic Signals Drawing (inside box)
# ----------------------------
def draw_signal_status(frame, green_lane, phase):
    coords = {"A": (100,50), "B": (300,50), "C": (500,50), "D": (700,50)}
    for lane, (x,y) in coords.items():
        # Draw box around signal
        cv2.rectangle(frame,(x-50,y-50),(x+50,y+50),(30,30,30),-1)
        color = (0,0,255)  # Red default
        if lane == green_lane and phase=="green": color=(0,255,0)
        elif lane == green_lane and phase=="yellow": color=(0,255,255)
        cv2.circle(frame,(x,y),30,color,-1)
        cv2.putText(frame,lane,(x-15,y+60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
    return frame

# ----------------------------
# Simulation Loop
# ----------------------------
if run_btn:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Vehicle detection
        lane_counts, emergency_detected, annotated_frame = detector.detect_vehicles(frame)

        # Decide green lane
        if override_lane!="None" and apply_override:
            green_lane, phase = override_lane, override_phase
        else:
            if use_rl:
                state = np.array([lane_counts[l] for l in ["A","B","C","D"]])
                action,_ = rl_model.predict(state, deterministic=True)
                green_lane = ["A","B","C","D"][action]
                phase="green"
            else:
                for lane,count in lane_counts.items(): controller.update_lane(lane,count)
                green_lane, phase = controller.decide_next(emergency_detected)

        # Draw signals inside boxes
        display_frame = draw_signal_status(annotated_frame, green_lane, phase)
        display_frame = cv2.resize(display_frame,(800,450))
        frame_placeholder.image(display_frame, channels="BGR")

        # Update metrics
        for lane in ["A","B","C","D"]:
            metric_objects[lane].metric(f"Lane {lane}", lane_counts[lane])

        # Congestion display
        congestion_html = ""
        for lane,count in lane_counts.items():
            if count<5: cls="congestion-low"
            elif count<12: cls="congestion-medium"
            else: cls="congestion-high"
            congestion_html += f"<span class='{cls}'>Lane {lane}: {count}</span> &nbsp; "
        st.markdown(congestion_html, unsafe_allow_html=True)

        # Append history
        st.session_state.history.append({
            "time": time.strftime("%H:%M:%S"),
            "A": lane_counts["A"], "B": lane_counts["B"], 
            "C": lane_counts["C"], "D": lane_counts["D"],
            "green": green_lane, "phase": phase,
            "emergency": emergency_detected
        })

        # Update chart with dynamic line colors based on congestion
        df = pd.DataFrame(st.session_state.history)
        if not df.empty:
            smoothed_df = df[["A","B","C","D"]].rolling(window=5,min_periods=1).mean()

            # Create custom color per lane based on congestion
            lane_colors = []
            latest = smoothed_df.iloc[-1]
            for lane in ["A","B","C","D"]:
                if latest[lane]>=12: lane_colors.append("red")
                elif latest[lane]>=5: lane_colors.append("orange")
                else: lane_colors.append("green")

            chart_placeholder.line_chart(smoothed_df, height=300, use_container_width=True)
            st.markdown("**Lane Colors:** A=Green, B=Blue, C=Orange, D=Red (dynamic congestion shading)")

            # Recent logs
            log_placeholder.dataframe(df.tail(20).sort_index(ascending=False), use_container_width=True)

    cap.release()
    st.success("✅ Simulation Finished")
