# 🐾 PawGuard — Dog Rescue AI System

An AI-powered stray dog detection, mapping, and rescue coordination system built with **Streamlit**, **Folium**, and **TensorFlow**.

---

## 📌 Project Overview

PawGuard helps detect injured or sick stray dogs using AI image recognition, pins their location on a real interactive map, and sends alerts to nearby rescue volunteers and NGOs.

### How It Works

1. **Upload a photo** of a stray dog
2. **AI analyses** the image and classifies the dog as Injured / Sick / Healthy
3. **Location is pinned** on a real OpenStreetMap
4. **Alert is sent** to nearby volunteers (Street Cause / NGOs)
5. **Rescue status** is tracked from Reported → Rescued → Treatment Done

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🤖 AI Image Classifier | Detects dog condition using pixel analysis / TensorFlow model |
| 📍 Real Interactive Map | Folium + OpenStreetMap with color-coded pins |
| 🚨 Alert System | Notifies volunteers via email / SMS (smtplib / Twilio) |
| 📋 Rescue Tracker | Status pipeline: Reported → Assigned → Rescued → Completed |
| 📊 Statistics Dashboard | Charts, metrics, and CSV export |

---

## 🗂️ Project Structure

```
pawguard/
├── app.py                  ← Main Streamlit app (all 5 pages)
├── model.py                ← AI model: training + prediction
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── pawguard.html           ← Standalone browser preview (no install needed)
└── pawguard_reports.csv    ← Auto-created when app runs (report data)
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone or download the project
```bash
# If using git:
git clone https://github.com/yourname/pawguard.git
cd pawguard

# Or just place all files in one folder
```

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ Python 3.9 or above recommended.

### Step 3 — (Optional) Train or download the AI model
```bash
# Train a new model using model.py:
python model.py

# This will create: dog_condition_model.h5
# Place it in the same folder as app.py
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 🤖 AI Model Details

The AI classifier in `model.py` uses **MobileNetV2** (transfer learning) to classify dog images into 3 categories:

| Label | Description |
|---|---|
| `Injured` | Wounds, bleeding, redness detected |
| `Sick` | Weak, lethargic, dull coat, malnutrition |
| `Healthy` | No visible signs of injury or illness |

### Dataset
- Collect dog images (100+ per class minimum)
- Organise into folders: `dataset/Injured/`, `dataset/Sick/`, `dataset/Healthy/`
- Run `python model.py` to train

### Replace Mock Classifier
In `app.py`, find the `classify_dog_image()` function and replace with:
```python
from model import predict_condition
result = predict_condition(image)
```

---

## 🗺️ Map Technology

- **Folium** (Python wrapper for Leaflet.js)
- **OpenStreetMap** tiles — free, no API key required
- **Color-coded pins**: 🔴 Injured · 🟠 Sick · 🟢 Healthy
- Click any pin to see full report details

---

## 🚨 Alert System Setup

### Email Alerts (Gmail)
In `app.py`, replace the `send_alert()` function:
```python
import smtplib
from email.mime.text import MIMEText

def send_alert(report):
    msg = MIMEText(f"New {report['condition']} dog at {report['location_name']}")
    msg['Subject'] = 'PawGuard Rescue Alert'
    msg['From'] = 'your@gmail.com'
    msg['To'] = 'volunteer@streetcause.org'
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login('your@gmail.com', 'app_password')
        server.send_message(msg)
```

### SMS Alerts (Twilio)
```python
from twilio.rest import Client

def send_alert(report):
    client = Client("ACCOUNT_SID", "AUTH_TOKEN")
    client.messages.create(
        body=f"PawGuard Alert: {report['condition']} dog at {report['location_name']}",
        from_="+1XXXXXXXXXX",
        to="+91XXXXXXXXXX"
    )
```

---

## 📊 Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | Stats overview + live map + recent reports |
| 📷 Report a Dog | Upload photo → AI analysis → submit report |
| 🗺️ Live Map | All reports on interactive map with filters |
| 📋 Rescue Tracker | Update rescue status per case |
| 📊 Statistics | Charts and CSV export |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Streamlit | Web UI framework |
| TensorFlow / Keras | AI image classification |
| MobileNetV2 | Pre-trained CNN backbone |
| OpenCV | Image preprocessing |
| Folium | Interactive map |
| Pandas | Data management |
| Pillow | Image loading |

---

## 👩‍💻 Developed By

**Varsha** — Dog Rescue AI Project  
Built as a machine learning project to help stray dog rescue organizations in Hyderabad, India.

---

## 📄 License

This project is open-source and free to use for animal welfare purposes.

> 🐕 *Every dog deserves a safe life.*
