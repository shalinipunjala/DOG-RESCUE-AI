"""
PawGuard — Dog Rescue AI System
================================
Single-file Streamlit app.

Install dependencies:
    pip install streamlit folium streamlit-folium pandas pillow numpy

Run:
    streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import json
import os
import io
import base64

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PawGuard — Dog Rescue AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; color: #fff; }
    .main-header p  { color: #a0b4c8; margin: 0.3rem 0 0; font-size: 1rem; }

    .stat-card {
        background: white;
        border: 1px solid #e8edf2;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stat-number { font-size: 2rem; font-weight: 600; color: #0f3460; }
    .stat-label  { font-size: 0.85rem; color: #6b7a8d; margin-top: 4px; }

    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .badge-reported   { background: #fff3cd; color: #856404; }
    .badge-assigned   { background: #cce5ff; color: #004085; }
    .badge-rescued    { background: #d4edda; color: #155724; }
    .badge-treatment  { background: #f8d7da; color: #721c24; }
    .badge-completed  { background: #d1ecf1; color: #0c5460; }

    .alert-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e8edf2;
    }
    div[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebar"] * { color: #c9d8e8 !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #a0b4c8 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA STORAGE  (in-session CSV mock)
# ──────────────────────────────────────────────
DATA_FILE = "pawguard_reports.csv"
COLUMNS = [
    "id", "timestamp", "location_name", "lat", "lon",
    "condition", "confidence", "breed_guess",
    "reporter_name", "reporter_phone",
    "status", "volunteer", "notes"
]

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    # Seed with sample data
    sample = pd.DataFrame([
        ["RPT001", "2026-04-06 09:12", "Jubilee Hills, Hyderabad", 17.4325, 78.4051,
         "Injured", 91, "Indian Pariah", "Ravi Kumar", "9876543210",
         "Volunteer Assigned", "Street Cause Team A", "Limping on left leg"],
        ["RPT002", "2026-04-07 14:45", "Banjara Hills, Hyderabad", 17.4156, 78.4483,
         "Sick", 84, "Mixed Breed", "Priya S", "9845012345",
         "Rescued", "Street Cause Team B", "Mange infection suspected"],
        ["RPT003", "2026-04-07 18:30", "Madhapur, Hyderabad", 17.4489, 78.3905,
         "Healthy", 78, "Labrador Mix", "Anon", "",
         "Reported", "", "Friendly stray, no visible injury"],
        ["RPT004", "2026-04-08 08:00", "Gachibowli, Hyderabad", 17.4401, 78.3489,
         "Injured", 95, "Indian Pariah", "Sai T", "8801234567",
         "Treatment Completed", "Dr. Meena Vet Clinic", "Fracture treated successfully"],
    ], columns=COLUMNS)
    sample.to_csv(DATA_FILE, index=False)
    return sample

def save_report(report_dict):
    df = load_data()
    new_row = pd.DataFrame([report_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def update_status(report_id, new_status, volunteer="", notes=""):
    df = load_data()
    mask = df["id"] == report_id
    df.loc[mask, "status"] = new_status
    if volunteer:
        df.loc[mask, "volunteer"] = volunteer
    if notes:
        df.loc[mask, "notes"] = notes
    df.to_csv(DATA_FILE, index=False)

# ──────────────────────────────────────────────
# AI CLASSIFIER  (mock — replace with real model)
# ──────────────────────────────────────────────
def classify_dog_image(image: Image.Image):
    """
    Mock AI classifier.
    Replace this function body with your TensorFlow/PyTorch model:

        model = tf.keras.models.load_model("model/dog_condition_model.h5")
        img_array = tf.keras.preprocessing.image.img_to_array(image.resize((224,224)))
        img_array = tf.expand_dims(img_array, 0) / 255.0
        predictions = model.predict(img_array)
        ...

    For now we analyze basic image statistics to simulate AI.
    """
    img_array = np.array(image.convert("RGB"))
    brightness = img_array.mean()
    red_ratio  = img_array[:,:,0].mean() / (img_array.mean() + 1)

    # Simulated heuristic (replace with real model output)
    if red_ratio > 1.12:
        condition   = "Injured"
        confidence  = int(min(95, 75 + red_ratio * 10))
        description = "Possible wounds or redness detected in the image."
    elif brightness < 90:
        condition   = "Sick"
        confidence  = int(min(90, 60 + (90 - brightness) * 0.4))
        description = "Low vitality indicators detected — dog may be weak or ill."
    else:
        condition   = "Healthy"
        confidence  = int(min(92, 70 + brightness * 0.1))
        description = "No visible signs of injury or sickness detected."

    breeds = ["Indian Pariah", "Mixed Breed", "Labrador Mix", "Street Indie", "Unknown Breed"]
    breed = breeds[int(brightness) % len(breeds)]

    return {
        "condition":   condition,
        "confidence":  confidence,
        "description": description,
        "breed_guess": breed,
    }

# ──────────────────────────────────────────────
# ALERT SYSTEM  (mock — replace with real email/SMS)
# ──────────────────────────────────────────────
def send_alert(report):
    """
    Mock alert sender.
    Replace with real implementation:

        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(f"New {report['condition']} dog reported at {report['location_name']}")
        msg['Subject'] = 'PawGuard Alert'
        msg['From'] = 'alerts@pawguard.org'
        msg['To'] = 'volunteers@streetcause.org'
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login('user', 'password')
            server.send_message(msg)

    For Twilio SMS:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        client.messages.create(body=msg_body, from_='+1234', to='+91...')
    """
    # Simulate network delay
    import time; time.sleep(0.8)
    return True

# ──────────────────────────────────────────────
# MAP BUILDER
# ──────────────────────────────────────────────
def build_map(df, center=(17.4326, 78.4071), zoom=12):
    m = folium.Map(location=center, zoom_start=zoom,
                   tiles="CartoDB positron")

    color_map = {
        "Injured":  "red",
        "Sick":     "orange",
        "Healthy":  "green",
    }
    icon_map = {
        "Reported":            "info-sign",
        "Volunteer Assigned":  "user",
        "Rescued":             "heart",
        "Treatment Completed": "ok-circle",
    }

    for _, row in df.iterrows():
        color = color_map.get(row["condition"], "blue")
        icon  = icon_map.get(row["status"], "question-sign")

        popup_html = f"""
        <div style='font-family:sans-serif;min-width:180px'>
          <b style='color:#0f3460'>{row['location_name']}</b><br>
          <span style='color:#555'>Condition:</span>
          <b style='color:{"crimson" if row["condition"]=="Injured" else "darkorange" if row["condition"]=="Sick" else "green"}'>{row['condition']}</b>
          ({row['confidence']}% confidence)<br>
          <span style='color:#555'>Breed:</span> {row['breed_guess']}<br>
          <span style='color:#555'>Status:</span> {row['status']}<br>
          <span style='color:#555'>Time:</span> {row['timestamp']}<br>
          <span style='color:#555'>Notes:</span> {row['notes']}<br>
        </div>
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['condition']} dog — {row['location_name']}",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:10px;
                border:1px solid #ddd;font-family:sans-serif;font-size:13px;'>
      <b>Dog Condition</b><br>
      <span style='color:crimson'>&#9679;</span> Injured&nbsp;&nbsp;
      <span style='color:darkorange'>&#9679;</span> Sick&nbsp;&nbsp;
      <span style='color:green'>&#9679;</span> Healthy
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐾 PawGuard")
    st.markdown("*Dog Rescue AI System*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Dashboard",
        "📷  Report a Dog",
        "🗺️  Live Map",
        "📋  Rescue Tracker",
        "📊  Statistics",
    ])
    st.markdown("---")
    st.markdown("**Emergency Contacts**")
    st.markdown("Street Cause: `040-1234-5678`")
    st.markdown("CUPA Hyderabad: `080-2222-3333`")
    st.markdown("---")
    st.caption("v1.0 · Built with Streamlit")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🐾 PawGuard — Dog Rescue AI</h1>
  <p>AI-powered stray dog detection, mapping & rescue coordination system</p>
</div>
""", unsafe_allow_html=True)

df = load_data()

# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
if page == "🏠  Dashboard":
    total     = len(df)
    injured   = len(df[df["condition"] == "Injured"])
    rescued   = len(df[df["status"].isin(["Rescued", "Treatment Completed"])])
    pending   = len(df[df["status"] == "Reported"])

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label, emoji in zip(
        [c1, c2, c3, c4],
        [total, injured, rescued, pending],
        ["Total Reports", "Injured Dogs", "Rescued", "Awaiting Response"],
        ["📌", "🚨", "💚", "⏳"]
    ):
        col.markdown(f"""
        <div class='stat-card'>
          <div style='font-size:1.6rem'>{emoji}</div>
          <div class='stat-number'>{num}</div>
          <div class='stat-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📍 Live Map Overview</div>", unsafe_allow_html=True)
    m = build_map(df)
    st_folium(m, width="100%", height=380)

    st.markdown("<div class='section-title'>🕐 Recent Reports</div>", unsafe_allow_html=True)
    recent = df.sort_values("timestamp", ascending=False).head(5)
    for _, row in recent.iterrows():
        badge_class = {
            "Reported": "badge-reported",
            "Volunteer Assigned": "badge-assigned",
            "Rescued": "badge-rescued",
            "Treatment Completed": "badge-completed",
        }.get(row["status"], "badge-reported")
        st.markdown(f"""
        <div style='background:white;border:1px solid #e8edf2;border-radius:10px;
                    padding:0.8rem 1.2rem;margin-bottom:0.5rem;display:flex;
                    justify-content:space-between;align-items:center;'>
          <div>
            <b>{row['location_name']}</b>
            <span style='color:#888;font-size:0.85rem;margin-left:8px'>{row['timestamp']}</span><br>
            <span style='color:#555;font-size:0.9rem'>{row['condition']} · {row['breed_guess']}</span>
          </div>
          <span class='status-badge {badge_class}'>{row['status']}</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: REPORT A DOG
# ══════════════════════════════════════════════
elif page == "📷  Report a Dog":
    st.markdown("<div class='section-title'>📷 Report a Stray Dog</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        uploaded_file = st.file_uploader(
            "Upload a photo of the dog",
            type=["jpg", "jpeg", "png"],
            help="Clear photo helps the AI assess the dog's condition accurately."
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded photo", use_container_width=True)

        st.markdown("**Location Details**")
        location_name = st.text_input("Area / Landmark", placeholder="e.g. Near Inorbit Mall, Madhapur")
        col_lat, col_lon = st.columns(2)
        lat = col_lat.number_input("Latitude",  value=17.4326, format="%.6f")
        lon = col_lon.number_input("Longitude", value=78.4071, format="%.6f")

        st.markdown("**Your Details** *(optional but helps volunteers)*")
        reporter_name  = st.text_input("Your Name")
        reporter_phone = st.text_input("Phone Number")
        extra_notes    = st.text_area("Additional Notes", placeholder="Describe what you saw...")

        submit = st.button("🚨 Submit Report & Send Alert", use_container_width=True, type="primary")

    with col_result:
        if uploaded_file and submit:
            with st.spinner("Analyzing image with AI..."):
                result = classify_dog_image(image)

            condition_colors = {"Injured": "#dc3545", "Sick": "#fd7e14", "Healthy": "#28a745"}
            color = condition_colors.get(result["condition"], "#0f3460")

            st.markdown(f"""
            <div style='background:white;border:2px solid {color};border-radius:14px;
                        padding:1.5rem;margin-bottom:1rem;'>
              <div style='font-size:1.1rem;font-weight:600;color:{color}'>
                AI Detection Result
              </div>
              <div style='font-size:2rem;font-weight:700;color:{color};margin:8px 0'>
                {result['condition']}
              </div>
              <div style='font-size:0.9rem;color:#555'>
                Confidence: <b>{result['confidence']}%</b><br>
                Breed estimate: <b>{result['breed_guess']}</b><br><br>
                {result['description']}
              </div>
            </div>""", unsafe_allow_html=True)

            if location_name:
                new_id = f"RPT{len(df)+1:03d}"
                report = {
                    "id": new_id,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "location_name": location_name,
                    "lat": lat, "lon": lon,
                    "condition":   result["condition"],
                    "confidence":  result["confidence"],
                    "breed_guess": result["breed_guess"],
                    "reporter_name":  reporter_name or "Anonymous",
                    "reporter_phone": reporter_phone,
                    "status":    "Reported",
                    "volunteer": "",
                    "notes":     extra_notes,
                }
                save_report(report)

                with st.spinner("Sending alert to volunteers..."):
                    alert_sent = send_alert(report)

                st.markdown(f"""
                <div class='success-box'>
                  ✅ <b>Report submitted!</b> ID: <code>{new_id}</code><br>
                  📧 Alert sent to Street Cause volunteers.<br>
                  🗺️ Dog pinned on the live map.
                </div>""", unsafe_allow_html=True)

                mini_map = build_map(pd.DataFrame([report]), center=(lat, lon), zoom=15)
                st_folium(mini_map, width="100%", height=200)
            else:
                st.warning("Please enter the location name before submitting.")

        elif not uploaded_file:
            st.markdown("""
            <div style='background:#f8f9fa;border-radius:12px;padding:2rem;text-align:center;color:#6b7a8d;'>
              <div style='font-size:3rem'>📸</div>
              <div style='margin-top:0.5rem'>Upload a photo to get AI analysis</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: LIVE MAP
# ══════════════════════════════════════════════
elif page == "🗺️  Live Map":
    st.markdown("<div class='section-title'>🗺️ Live Map — All Reported Dogs</div>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    filter_condition = col_f1.multiselect("Filter by Condition",
        ["Injured", "Sick", "Healthy"], default=["Injured", "Sick", "Healthy"])
    filter_status = col_f2.multiselect("Filter by Status",
        df["status"].unique().tolist(), default=df["status"].unique().tolist())
    col_f3.markdown("<br>", unsafe_allow_html=True)
    col_f3.metric("Showing", f"{len(df[df['condition'].isin(filter_condition) & df['status'].isin(filter_status)])} dogs")

    filtered = df[df["condition"].isin(filter_condition) & df["status"].isin(filter_status)]
    m = build_map(filtered)
    st_folium(m, width="100%", height=520)

# ══════════════════════════════════════════════
# PAGE: RESCUE TRACKER
# ══════════════════════════════════════════════
elif page == "📋  Rescue Tracker":
    st.markdown("<div class='section-title'>📋 Rescue Status Tracker</div>", unsafe_allow_html=True)

    df = load_data()
    status_order = ["Reported", "Volunteer Assigned", "Rescued", "Treatment Completed"]

    for status in status_order:
        group = df[df["status"] == status]
        badge_class = {
            "Reported": "badge-reported",
            "Volunteer Assigned": "badge-assigned",
            "Rescued": "badge-rescued",
            "Treatment Completed": "badge-completed",
        }.get(status, "badge-reported")

        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.5rem'>
          <span class='status-badge {badge_class}' style='font-size:0.9rem;padding:5px 14px'>
            {status}
          </span>
          <span style='color:#888;font-size:0.85rem'>{len(group)} case(s)</span>
        </div>""", unsafe_allow_html=True)

        if group.empty:
            st.caption("  No cases in this stage.")
        else:
            for _, row in group.iterrows():
                with st.expander(f"📍 {row['location_name']} — {row['condition']} ({row['id']})"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"""
                    **Report ID:** `{row['id']}`
                    **Timestamp:** {row['timestamp']}
                    **Location:** {row['location_name']}
                    **Condition:** {row['condition']} ({row['confidence']}% confidence)
                    **Breed:** {row['breed_guess']}
                    """)
                    c2.markdown(f"""
                    **Reporter:** {row['reporter_name']}
                    **Phone:** {row['reporter_phone'] or 'N/A'}
                    **Volunteer:** {row['volunteer'] or 'Not assigned'}
                    **Notes:** {row['notes']}
                    """)

                    st.markdown("**Update Status:**")
                    new_status    = st.selectbox("New Status", status_order,
                                                  index=status_order.index(status),
                                                  key=f"status_{row['id']}")
                    new_volunteer = st.text_input("Assign Volunteer / Org", value=row['volunteer'],
                                                   key=f"vol_{row['id']}")
                    new_notes     = st.text_area("Update Notes", value=row['notes'],
                                                  key=f"notes_{row['id']}")
                    if st.button("💾 Save Update", key=f"save_{row['id']}"):
                        update_status(row['id'], new_status, new_volunteer, new_notes)
                        st.success("Status updated!")
                        st.rerun()

# ══════════════════════════════════════════════
# PAGE: STATISTICS
# ══════════════════════════════════════════════
elif page == "📊  Statistics":
    st.markdown("<div class='section-title'>📊 Rescue Statistics</div>", unsafe_allow_html=True)

    df = load_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Reports by Condition**")
        cond_counts = df["condition"].value_counts()
        st.bar_chart(cond_counts)

        st.markdown("**Reports by Status**")
        status_counts = df["status"].value_counts()
        st.bar_chart(status_counts)

    with col2:
        st.markdown("**Top Locations with Reports**")
        loc_counts = df["location_name"].value_counts().head(8)
        st.bar_chart(loc_counts)

        st.markdown("**AI Confidence Distribution**")
        st.area_chart(df[["confidence"]])

    st.markdown("---")
    st.markdown("**Full Data Table**")
    st.dataframe(
        df[["id", "timestamp", "location_name", "condition", "confidence",
            "breed_guess", "status", "volunteer"]].sort_values("timestamp", ascending=False),
        use_container_width=True
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Report (CSV)", csv,
                       "pawguard_reports.csv", "text/csv")
