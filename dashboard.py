import streamlit as st
import cv2
import pandas as pd
import xgboost as xgb
import numpy as np
import base64
import os
from datetime import datetime
from detector import ShopliftingDetector
import streamlit.components.v1 as components

# ─── Helpers ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

def put_label(img, text, pos, color_bg, color_text=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 8, y + baseline), color_bg, -1)
    cv2.putText(img, text, (x + 4, y), font, scale, color_text, thick, cv2.LINE_AA)

@st.cache_data
def load_alert_sound_b64():
    try:
        path = os.path.join(BASE_DIR, "alertsound.wav")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

def inject_audio_controller():
    """Inject a persistent looping audio player into the parent window."""
    b64 = load_alert_sound_b64()
    if b64 is None:
        return
    components.html(f"""
    <script>
    (function() {{
        var win = window.parent || window;
        var doc = win.document;
        if (doc.getElementById('rg_alarm_audio')) return;
        var audio = doc.createElement('audio');
        audio.id   = 'rg_alarm_audio';
        audio.loop = true;
        audio.src  = 'data:audio/wav;base64,{b64}';
        audio.volume = 1.0;
        doc.body.appendChild(audio);
        function tick() {{
            var state = win.localStorage.getItem('retailguard_alarm');
            if (state === 'play') {{
                if (audio.paused) audio.play().catch(function(e) {{ console.warn('Audio blocked:', e); }});
            }} else {{
                if (!audio.paused) {{ audio.pause(); audio.currentTime = 0; }}
            }}
        }}
        setInterval(tick, 200);
    }})();
    </script>
    """, height=0)

def set_alarm(state: str):
    components.html(f"""
    <script>
    (window.parent || window).localStorage.setItem('retailguard_alarm', '{state}');
    </script>
    """, height=0)

def save_capture(frame, frame_num, prob):
    """Save a suspicious frame as JPG and return the file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"suspect_{ts}_f{frame_num}_s{prob:.2f}.jpg"
    path = os.path.join(CAPTURES_DIR, filename)
    success = cv2.imwrite(path, frame)
    if not success:
        print(f"Failed to save image to {path}") # Visible in terminal console
    return path

def img_to_b64(path):
    """Convert image file to base64 for inline HTML display."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailGuard – Surveillance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a0e1a 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
.dashboard-header {
    background: linear-gradient(90deg, #1e3a5f 0%, #1a2744 50%, #0f172a 100%);
    border: 1px solid #2563eb33; border-radius: 14px;
    padding: 22px 30px; margin-bottom: 22px;
    display: flex; align-items: center; gap: 18px;
    box-shadow: 0 4px 28px rgba(37,99,235,0.18);
}
.dashboard-header h1 {
    margin: 0; font-size: 1.85rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.dashboard-header p { margin: 4px 0 0; color: #94a3b8; font-size: 0.85rem; }

.source-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid #334155; border-radius: 14px;
    padding: 28px 20px; text-align: center; transition: all 0.25s ease;
}
.source-card:hover {
    border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.2);
    transform: translateY(-3px);
}
.source-card .icon  { font-size: 2.4rem; margin-bottom: 10px; }
.source-card .title { font-size: 1rem; font-weight: 600; color: #e2e8f0; }
.source-card .desc  { font-size: 0.78rem; color: #64748b; margin-top: 4px; }

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 18px 20px; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
.metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 6px; }
.metric-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }

.alert-item {
    background: linear-gradient(90deg, #450a0a 0%, #1a0505 100%);
    border-left: 4px solid #ef4444; border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin-bottom: 8px;
    font-size: 0.82rem; color: #fca5a5;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* Capture gallery */
.capture-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px;
    margin-top: 10px;
}
.capture-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid #ef444455;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(239,68,68,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.capture-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(239,68,68,0.3);
    border-color: #ef4444;
}
.capture-card img {
    width: 100%; display: block;
    border-bottom: 1px solid #ef444433;
}
.capture-card .cap-info {
    padding: 8px 10px;
    font-size: 0.72rem; color: #f87171;
    font-weight: 500;
}
.capture-card .cap-score {
    padding: 0 10px 10px 10px;
    font-size: 0.68rem; color: #64748b; margin-top: 2px;
}

.status-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: #052e16; border: 1px solid #16a34a;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; font-weight: 600; color: #4ade80;
}
.status-idle {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e293b; border: 1px solid #475569;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; font-weight: 600; color: #94a3b8;
}
.section-title {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}
.source-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e3a5f; border: 1px solid #2563eb55;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; color: #93c5fd;
    margin-bottom: 12px;
}
div.stButton > button {
    width: 100%; border-radius: 8px; font-weight: 600;
    font-size: 0.9rem; padding: 10px; transition: all 0.2s ease;
}
#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
defaults = {
    'running': False,
    'alerts': [],
    'suspicious_count': 0,
    'normal_count': 0,
    'frames_processed': 0,
    'detector': None,
    'source_mode': None,
    'rtsp_url': '',
    'webcam_index': 0,
    'alarm_active': False,
    'captures': [],   # list of dicts: {path, time, frame, score}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Inject persistent audio controller ─────────────────────────────────────────
inject_audio_controller()

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div style="font-size:2.6rem;">&#128737;</div>
    <div>
        <h1>RetailGuard Surveillance Dashboard</h1>
        <p>AI-powered shoplifting detection &nbsp;&bull;&nbsp; YOLO Pose + XGBoost &nbsp;&bull;&nbsp; Video File &middot; RTSP &middot; Webcam</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("---")

    conf_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.55, 0.05,
                               help="Minimum YOLO pose confidence")
    sus_threshold  = st.slider("Suspicion Threshold",  0.3, 0.9, 0.5,  0.05,
                               help="XGBoost probability below this = Suspicious")
    sound_enabled  = st.checkbox("Alert Sound", value=True,
                                 help="Loop alertsound.wav while suspicious behavior is detected")
    max_captures   = st.slider("Max Captures Shown", 6, 50, 20, 2,
                               help="Maximum number of shoplifter snapshots to display")

    st.markdown("---")
    # Start/Stop buttons are shown in the main area (see below)
    start_btn = False  # placeholder, defined in main area
    stop_btn  = False  # placeholder, defined in main area

    st.markdown("---")
    st.markdown("### Session Log")
    log_placeholder = st.empty()

    st.markdown("---")
    if st.button("Clear Captures", use_container_width=True):
        st.session_state.captures = []
        try:
            for f in os.listdir(CAPTURES_DIR):
                os.remove(os.path.join(CAPTURES_DIR, f))
        except Exception:
            pass
        st.rerun()

    st.markdown(
        "<div style='color:#475569; font-size:0.72rem; text-align:center; margin-top:8px;'>"
        "RetailGuard v1.0 &bull; YOLOv11 + XGBoost</div>",
        unsafe_allow_html=True
    )

# ─── Source Picker ───────────────────────────────────────────────────────────────
if st.session_state.source_mode is None:
    st.markdown("""
    <div style='text-align:center; margin: 10px 0 4px;'>
        <h2 style='color:#e2e8f0; font-weight:700;'>Select a Video Source</h2>
        <p style='color:#64748b; font-size:0.9rem;'>Choose how you want to feed video into the detection system</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="source-card">
            <div class="icon">&#127909;</div>
            <div class="title">Video File</div>
            <div class="desc">Analyse a pre-recorded .mp4 file from the project folder</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Use Video File", key="btn_file", use_container_width=True):
            st.session_state.source_mode = 'file'; st.rerun()

    with c2:
        st.markdown("""<div class="source-card">
            <div class="icon">&#128225;</div>
            <div class="title">RTSP Stream</div>
            <div class="desc">Connect to an IP camera or NVR via an RTSP URL</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Use RTSP Stream", key="btn_rtsp", use_container_width=True):
            st.session_state.source_mode = 'rtsp'; st.rerun()

    with c3:
        st.markdown("""<div class="source-card">
            <div class="icon">&#128247;</div>
            <div class="title">Webcam</div>
            <div class="desc">Use a locally connected USB or built-in webcam</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Use Webcam", key="btn_webcam", use_container_width=True):
            st.session_state.source_mode = 'webcam'; st.rerun()

    st.stop()

# ─── Source Configuration ────────────────────────────────────────────────────────
mode = st.session_state.source_mode

# Set defaults to avoid ReferenceError if mode is unexpected
badge_icon, badge_text, cv_source = "&#10067;", "Unknown", 0

if mode == 'file':
    badge_icon, badge_text = "&#127909;", "Video File"
    video_file = st.selectbox("Select Video File", ["vid.mp4", "nm1.mp4", "susup1.mp4"])
    cv_source  = video_file

elif mode == 'rtsp':
    badge_icon, badge_text = "&#128225;", "RTSP Stream"
    st.markdown('<div class="section-title">RTSP Connection</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([4, 1])
    with r1:
        rtsp_input = st.text_input("RTSP URL", value=st.session_state.rtsp_url,
                                   placeholder="rtsp://user:pass@192.168.1.100:554/stream")
    with r2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Test", use_container_width=True):
            with st.spinner("Testing..."):
                tc = cv2.VideoCapture(rtsp_input)
                st.success("Connected!") if tc.isOpened() else st.error("Cannot connect")
                tc.release()
    st.session_state.rtsp_url = rtsp_input
    cv_source = rtsp_input

elif mode == 'webcam':
    badge_icon, badge_text = "&#128247;", "Webcam"
    cam_idx = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
    st.session_state.webcam_index = cam_idx
    cv_source = int(cam_idx)

col_badge, col_start_main, col_stop_main, col_change = st.columns([3, 1, 1, 1])
with col_badge:
    st.markdown(f'<div class="source-badge">{badge_icon} Active source: <strong>{badge_text}</strong></div>',
                unsafe_allow_html=True)
with col_start_main:
    start_btn = st.button("▶ Start", use_container_width=True, type="primary")
with col_stop_main:
    stop_btn = st.button("⏹ Stop", use_container_width=True)
with col_change:
    if st.button("🔄 Change", use_container_width=True):
        st.session_state.source_mode = None
        st.session_state.running = False
        set_alarm('stop')
        st.rerun()

if stop_btn:
    st.session_state.running = False
    set_alarm('stop')

# ─── Metrics Row ────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1: suspicious_metric = st.empty()
with m2: normal_metric     = st.empty()
with m3: frames_metric     = st.empty()
with m4: status_metric     = st.empty()

def render_metrics(sus, norm, frames, is_live):
    suspicious_metric.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#ef4444;">&#128680; {sus}</div>
        <div class="metric-label">Suspicious Events</div></div>""", unsafe_allow_html=True)
    normal_metric.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#22c55e;">&#9989; {norm}</div>
        <div class="metric-label">Normal Detections</div></div>""", unsafe_allow_html=True)
    frames_metric.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#60a5fa;">&#127902; {frames}</div>
        <div class="metric-label">Frames Processed</div></div>""", unsafe_allow_html=True)
    status_metric.markdown(
        '<div class="metric-card"><div class="status-live">&#128308; LIVE</div>'
        '<div class="metric-label" style="margin-top:8px;">System Status</div></div>'
        if is_live else
        '<div class="metric-card"><div class="status-idle">&#9898; IDLE</div>'
        '<div class="metric-label" style="margin-top:8px;">System Status</div></div>',
        unsafe_allow_html=True
    )

render_metrics(st.session_state.suspicious_count, st.session_state.normal_count,
               st.session_state.frames_processed, st.session_state.running)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Video + Alerts ─────────────────────────────────────────────────────────────
vid_col, alert_col = st.columns([3, 1])

with vid_col:
    st.markdown('<div class="section-title">&#128247; Live Video Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    video_placeholder.markdown(
        "<div style='background:#0a0e1a; border:1px solid #1e293b; border-radius:12px;"
        "height:400px; display:flex; align-items:center; justify-content:center;"
        "color:#334155; font-size:1.1rem;'>Press Start to begin surveillance</div>",
        unsafe_allow_html=True
    )

with alert_col:
    st.markdown('<div class="section-title">&#128680; Real-time Alerts</div>', unsafe_allow_html=True)
    alert_placeholder = st.empty()
    alert_placeholder.markdown(
        "<div style='color:#475569; font-size:0.82rem; padding:12px;'>No alerts yet...</div>",
        unsafe_allow_html=True
    )

# ─── Captures Gallery ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">&#128247; Captured Shoplifters</div>', unsafe_allow_html=True)
gallery_placeholder = st.empty()

def render_gallery(captures, max_n):
    if not captures:
        gallery_placeholder.markdown(
            "<div style='color:#475569; font-size:0.85rem; padding:12px;'>"
            "No suspicious frames captured yet. Captures will appear here automatically.</div>",
            unsafe_allow_html=True
        )
        return

    shown = captures[:max_n]
    cards_html = '<div class="capture-gallery">'
    for cap in shown:
        try:
            b64 = img_to_b64(cap['path'])
            if b64:
                cards_html += (
                    f'<div class="capture-card">'
                    f'<img src="data:image/jpeg;base64,{b64}" alt="suspect"/>'
                    f'<div class="cap-info">&#128680; {cap["time"]}</div>'
                    f'<div class="cap-score">Frame #{cap["frame"]} score {cap["score"]:.2f}</div>'
                    f'</div>'
                )
        except Exception as e:
            print(f"Error rendering capture {cap['path']}: {e}")
            pass
    cards_html += '</div>'
    gallery_placeholder.markdown(cards_html, unsafe_allow_html=True)

render_gallery(st.session_state.captures, max_captures)

# ─── Detection Loop ──────────────────────────────────────────────────────────────
if start_btn:
    if mode == 'rtsp' and not st.session_state.rtsp_url.strip():
        st.error("Please enter an RTSP URL before starting.")
        st.stop()

    st.session_state.suspicious_count = 0
    st.session_state.normal_count     = 0
    st.session_state.frames_processed = 0
    st.session_state.alerts           = []
    # IMPORTANT: DO NOT clear captures list here, allowing it to persist across runs
    # st.session_state.captures       = [] 
    st.session_state.alarm_active     = False
    st.session_state.running          = True
    
    # Reload from disk if session state empty (optional, for restart)
    if not st.session_state.captures:
        try:
            files = [os.path.join(CAPTURES_DIR, f) for f in os.listdir(CAPTURES_DIR) if f.endswith('.jpg')]
            files.sort(key=os.path.getmtime, reverse=True)
            for f in files:
                # Parse filename to get metadata if possible, else defaults
                # suspect_YYYYMMDD_HHMMSS_f123_s0.99.jpg
                fname = os.path.basename(f)
                try:
                    parts = fname.split('_')
                    # parts: suspect, YYYYMMDD, HHMMSS, f123, s0.99.jpg
                    ts_str = f"{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:]}"
                    frm = int(parts[3][1:])
                    sc = float(parts[4][1:-4])
                    st.session_state.captures.append({'path': f, 'time': ts_str, 'frame': frm, 'score': sc})
                except:
                    st.session_state.captures.append({'path': f, 'time': 'Unknown', 'frame': 0, 'score': 0.0})
        except:
            pass

    if st.session_state.detector is None:
        with st.spinner("Loading AI models (YOLO + XGBoost)..."):
            try:
                st.session_state.detector = ShopliftingDetector()
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                st.stop()

    detector = st.session_state.detector

    with st.spinner(f"Connecting to {badge_text}..."):
        cap = cv2.VideoCapture(cv_source)

    if not cap.isOpened():
        st.error(f"Cannot open source: {cv_source}")
        st.session_state.running = False
        st.stop()

    last_capture_frame = -60  # Initial cooldown

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            if mode == 'file':
                st.session_state.running = False
                break
            else:
                cap.release()
                cap = cv2.VideoCapture(cv_source)
                continue

        frame = cv2.resize(frame, (1018, 600))
        results = detector.model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        frame_has_suspicious = False
        best_sus_score = 0.0

        for r in results:
            bound_box = r.boxes.xyxy
            conf      = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
                if conf[index] < conf_threshold:
                    continue

                x1, y1, x2, y2 = box.tolist()
                kp = keypoints[index]
                if len(kp) == 0:
                    continue

                cols, vals = [], []
                for j in range(len(kp)):
                    cols.append(f'x{j}'); vals.append(kp[j][0])
                    cols.append(f'y{j}'); vals.append(kp[j][1])

                df       = pd.DataFrame([vals], columns=cols)
                dmatrix  = xgb.DMatrix(df)
                sus_prob = detector.model.predict(dmatrix)
                prob_val = float(sus_prob[0]) if hasattr(sus_prob, '__len__') else float(sus_prob)
                pred     = 0 if prob_val < sus_threshold else 1

                if pred == 0:
                    frame_has_suspicious = True
                    conf_score = 1.0 - prob_val
                    if conf_score > best_sus_score:
                        best_sus_score = conf_score
                        
                    st.session_state.suspicious_count += 1
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    put_label(annotated_frame, "!! SUSPICIOUS", (int(x1), max(int(y1) - 4, 20)), (180, 0, 0))
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.alerts.insert(0, {
                        "time": ts, "label": "Suspicious",
                        "frame": st.session_state.frames_processed,
                        "conf": prob_val
                    })
                else:
                    st.session_state.normal_count += 1
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 80), 2)
                    put_label(annotated_frame, "Normal", (int(x1), max(int(y1) - 4, 20)), (0, 140, 60))

        # ─── Save capture ─────────────────────────────────────────────────────────
        cur_frame = st.session_state.frames_processed
        if frame_has_suspicious and (cur_frame - last_capture_frame) >= 45: # Capture every 45 frames (approx 1.5s) if still suspicious
            last_capture_frame = cur_frame
            try:
                path = save_capture(annotated_frame, cur_frame, best_sus_score)
                capture_entry = {
                    "path":  path,
                    "time":  datetime.now().strftime("%H:%M:%S"),
                    "frame": cur_frame,
                    "score": best_sus_score
                }
                st.session_state.captures.insert(0, capture_entry)
                
                # Force gallery refresh occasionally to show live updates
                render_gallery(st.session_state.captures, max_captures)
                
            except Exception as e:
                print(f"Capture error: {e}")

        # ─── Alarm control ─────────────────────────────────────────────────────────
        if sound_enabled:
            if frame_has_suspicious and not st.session_state.alarm_active:
                set_alarm('play')
                st.session_state.alarm_active = True
            elif not frame_has_suspicious and st.session_state.alarm_active:
                set_alarm('stop')
                st.session_state.alarm_active = False

        st.session_state.frames_processed += 1

        # Update video
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update metrics
        render_metrics(st.session_state.suspicious_count, st.session_state.normal_count,
                       st.session_state.frames_processed, True)

        # Update alerts panel
        alerts_html = "".join(
            f'<div class="alert-item"><strong>&#128680; {a["time"]}</strong><br>'
            f'Suspicious behavior detected<br>'
            f'<span style="color:#f87171; font-size:0.75rem;">'
            f'Frame #{a["frame"]} &nbsp;&bull;&nbsp; Score: {a["conf"]:.3f}</span></div>'
            for a in st.session_state.alerts[:15]
        ) or "<div style='color:#475569; font-size:0.82rem; padding:12px;'>No alerts yet...</div>"
        alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)

        # Sidebar log
        log_lines = "\n".join(f"[{a['time']}] Frame#{a['frame']}" for a in st.session_state.alerts[:8])
        log_placeholder.text(log_lines or "No events yet.")

    cap.release()
    set_alarm('stop')
    st.session_state.alarm_active = False
    st.session_state.running = False

    render_metrics(st.session_state.suspicious_count, st.session_state.normal_count,
                   st.session_state.frames_processed, False)
    render_gallery(st.session_state.captures, max_captures)
    st.success(f"Stopped. {len(st.session_state.captures)} Suspicious frames captured.")
