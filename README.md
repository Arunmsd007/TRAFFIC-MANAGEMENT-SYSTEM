# 🚦 Smart Traffic Management Dashboard

A real-time AI-based traffic management system that detects vehicles, monitors congestion, and optimizes traffic signal timings in urban areas. The dashboard provides live metrics, traffic signal visualization, vehicle counts per lane, congestion analysis, and event logs.

---

## **Features**

- **Real-time Vehicle Detection**  
  Detects cars, bikes, buses, and trucks using YOLOv8, and assigns them to Lane A/B/C/D based on horizontal position.

- **Traffic Signal Visualization**  
  Dynamic green/yellow/red signals displayed inside lane boxes.

- **Lane Vehicle Metrics**  
  Real-time metrics of vehicle count per lane with congestion coloring:
  - Green → Low (0–4 vehicles)
  - Orange → Medium (5–11 vehicles)
  - Red → High (12+ vehicles)

- **Traffic Density Graph**  
  Smoothed rolling average of vehicle counts per lane, updated in real-time with congestion warnings.

- **Event Logs**  
  Shows last 20 frames with time, lane counts, green lane, phase, and emergency vehicle detection.

- **Manual Override & RL Controller**  
  Optional reinforcement learning (DQN) or manual control for lane signals.

- **Error Handling**  
  Safely handles missing video files or incorrect paths.

---



---

## **Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/yourusername/smart-traffic-dashboard.git
cd smart-traffic-dashboard


2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt


Example requirements.txt:

streamlit
opencv-python
pandas
numpy
ultralytics
stable-baselines3

4. Download YOLOv8 Weights
# Download yolov8n.pt from Ultralytics


Usage
streamlit run app.py


Use the sidebar to:

Select video path

Enable RL or manual controller

Apply manual lane overrides

The dashboard displays:

Live video feed with annotated vehicles

Traffic signals in boxes

Lane metrics

Traffic density graph

Event logs

Traffic Signal Rules

Green lane: Vehicles can move

Yellow lane: Prepare to stop

Red lane: Stop

Emergency vehicle (Bus): Prioritized lane switching

Future Improvements

Dynamic lane lines on the graph based on congestion

Integration with live camera feeds

Predictive signal control using ML/RL

Mobile-friendly responsive UI

References

Ultralytics YOLOv8

Streamlit Documentation

Stable-Baselines3 (RL)
