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
    # Prefer tamil.mp3; fall back to alertsound.wav
    for fname, mime in [("tamil.mp3", "audio/mpeg"), ("alertsound.wav", "audio/wav")]:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode(), mime
    return None, None

def inject_audio_controller():
    b64, mime = load_alert_sound_b64()
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
        audio.src  = 'data:{mime};base64,{b64}';
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"suspect_{ts}_f{frame_num}_s{prob:.2f}.jpg"
    path = os.path.join(CAPTURES_DIR, filename)
    cv2.imwrite(path, frame)
    return path

def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailGuard ",
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
section[data-testid="stSidebar"] { display: none; }

.dashboard-header {
    background: linear-gradient(90deg, #1e3a5f 0%, #1a2744 50%, #0f172a 100%);
    border: 1px solid #2563eb33; border-radius: 14px;
    padding: 18px 26px; margin-bottom: 18px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 4px 28px rgba(37,99,235,0.18);
}
.dashboard-header h1 {
    margin: 0; font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.dashboard-header p { margin: 3px 0 0; color: #94a3b8; font-size: 0.8rem; }

/* ── Left Control Panel ── */
.ctrl-panel {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 18px 16px;
    height: 100%;
}
.ctrl-section {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #475569;
    margin: 14px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #1e293b;
}
.source-badge-sm {
    display: inline-flex; align-items: center; gap: 5px;
    background: #1e3a5f; border: 1px solid #2563eb55;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.74rem; font-weight: 600; color: #93c5fd;
    margin-bottom: 10px; width: 100%; justify-content: center;
}
.alert-item {
    background: linear-gradient(90deg, #450a0a 0%, #1a0505 100%);
    border-left: 3px solid #ef4444; border-radius: 0 6px 6px 0;
    padding: 8px 10px; margin-bottom: 6px;
    font-size: 0.75rem; color: #fca5a5;
    animation: slideIn 0.3s ease;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}
.alert-scroll {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 2px;
}
.alert-scroll::-webkit-scrollbar { width: 4px; }
.alert-scroll::-webkit-scrollbar-track { background: #0a0e1a; }
.alert-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 14px 16px; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 1.9rem; font-weight: 700; line-height: 1; margin-bottom: 5px; }
.metric-label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }

.status-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: #052e16; border: 1px solid #16a34a;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; font-weight: 600; color: #4ade80;
}
.status-idle {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e293b; border: 1px solid #475569;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; font-weight: 600; color: #94a3b8;
}
.section-title {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b;
    margin-bottom: 10px; padding-bottom: 5px;
    border-bottom: 1px solid #1e293b;
}
/* ── Gallery ── */
.capture-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px; margin-top: 8px;
}
.capture-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid #ef444455; border-radius: 10px;
    overflow: hidden; box-shadow: 0 4px 14px rgba(239,68,68,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.capture-card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(239,68,68,0.3); border-color: #ef4444; }
.capture-card img { width: 100%; display: block; border-bottom: 1px solid #ef444433; }
.capture-card .cap-info { padding: 6px 10px; font-size: 0.7rem; color: #f87171; font-weight: 500; }
.capture-card .cap-score { padding: 0 10px 8px; font-size: 0.65rem; color: #64748b; }

/* Source picker cards */
.source-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid #334155; border-radius: 12px;
    padding: 20px 14px; text-align: center; transition: all 0.25s ease;
}
.source-card:hover { border-color: #3b82f6; transform: translateY(-3px); }
.source-card .icon  { font-size: 2rem; margin-bottom: 8px; }
.source-card .title { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
.source-card .desc  { font-size: 0.72rem; color: #64748b; margin-top: 3px; }

div.stButton > button {
    width: 100%; border-radius: 8px; font-weight: 600;
    font-size: 0.88rem; padding: 9px; transition: all 0.2s ease;
}
#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}
[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
defaults = {
    'running': False, 'alerts': [],
    'suspicious_count': 0, 'normal_count': 0,
    'frames_processed': 0, 'detector': None,
    'source_mode': None, 'rtsp_url': '',
    'webcam_index': 0, 'alarm_active': False,
    'captures': [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

inject_audio_controller()

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div style="font-size:2.2rem;">&#128737;</div>
    <div>
        <h1>Automated Shoplifting Detection & Alert System</h1>
        <p>AI-powered shoplifting detection &nbsp;&bull;&nbsp; YOLO Pose + XGBoost &nbsp;&bull;&nbsp; Video File &middot; RTSP &middot; Webcam</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Source Picker (full-page, shown only when no source selected) ───────────────
if st.session_state.source_mode is None:
    st.markdown("""
    <div style='text-align:center; margin: 16px 0 6px;'>
        <h2 style='color:#e2e8f0; font-weight:700;'>Select a Video Source</h2>
        <p style='color:#64748b; font-size:0.88rem;'>Choose how you want to feed video into the detection system</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="source-card">
            <div class="icon">&#127909;</div>
            <div class="title">Video File</div>
            <div class="desc">Analyse a pre-recorded .mp4 file</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Use Video File", key="btn_file", use_container_width=True):
            st.session_state.source_mode = 'file'; st.rerun()
    with c2:
        st.markdown("""<div class="source-card">
            <div class="icon">&#128225;</div>
            <div class="title">RTSP Stream</div>
            <div class="desc">Connect to an IP camera via RTSP URL</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Use RTSP Stream", key="btn_rtsp", use_container_width=True):
            st.session_state.source_mode = 'rtsp'; st.rerun()
    with c3:
        st.markdown("""<div class="source-card">
            <div class="icon">&#128247;</div>
            <div class="title">Webcam</div>
            <div class="desc">Use a locally connected webcam</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Use Webcam", key="btn_webcam", use_container_width=True):
            st.session_state.source_mode = 'webcam'; st.rerun()
    st.stop()

# ─── Source Config ───────────────────────────────────────────────────────────────
mode = st.session_state.source_mode
badge_icon, badge_text, cv_source = "&#127909;", "Video File", "vid.mp4"

# ─── MAIN LAYOUT: Left Panel | Right Video ───────────────────────────────────────
left_col, right_col = st.columns([1, 3])

# ════════════════════════════════════════════
# LEFT PANEL
# ════════════════════════════════════════════
with left_col:
    st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)

    # ── Source Config inside left panel ──
    st.markdown('<div class="ctrl-section">📡 Video Source</div>', unsafe_allow_html=True)

    if mode == 'file':
        badge_icon, badge_text = "&#127909;", "Video File"
        video_file = st.selectbox("Select File", ["vid.mp4", "nm1.mp4", "susup1.mp4", "tamil.mp4"], label_visibility="collapsed")
        cv_source = video_file

    elif mode == 'rtsp':
        badge_icon, badge_text = "&#128225;", "RTSP Stream"
        rtsp_input = st.text_input("RTSP URL", value=st.session_state.rtsp_url,
                                   placeholder="rtsp://user:pass@ip:554/stream",
                                   label_visibility="collapsed")
        if st.button("🔌 Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                tc = cv2.VideoCapture(rtsp_input)
                st.success("Connected!") if tc.isOpened() else st.error("Cannot connect")
                tc.release()
        st.session_state.rtsp_url = rtsp_input
        cv_source = rtsp_input

    elif mode == 'webcam':
        badge_icon, badge_text = "&#128247;", "Webcam"
        cam_idx = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1, label_visibility="collapsed")
        st.session_state.webcam_index = cam_idx
        cv_source = int(cam_idx)

    st.markdown(f'<div class="source-badge-sm">{badge_icon}&nbsp;{badge_text}</div>', unsafe_allow_html=True)

    if st.button("🔄 Change Source", use_container_width=True):
        st.session_state.source_mode = None
        st.session_state.running = False
        set_alarm('stop')
        st.rerun()

    # ── Detection Controls ──
    st.markdown('<div class="ctrl-section">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence", 0.3, 0.9, 0.55, 0.05, help="YOLO pose confidence threshold")
    sus_threshold  = st.slider("Suspicion",  0.3, 0.9, 0.5,  0.05, help="XGBoost threshold — below = Suspicious")
    sound_enabled  = st.checkbox("🔔 Alert Sound", value=True)
    max_captures   = st.slider("Max Captures", 6, 50, 20, 2)

    # ── START / STOP Buttons ──
    st.markdown('<div class="ctrl-section">🎮 Controls</div>', unsafe_allow_html=True)
    start_btn = st.button("▶ START Detection", use_container_width=True, type="primary")
    stop_btn  = st.button("⏹ STOP Detection",  use_container_width=True)

    if stop_btn:
        st.session_state.running = False
        set_alarm('stop')

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Clear All Captures", use_container_width=True):
        st.session_state.captures = []
        try:
            for f in os.listdir(CAPTURES_DIR):
                os.remove(os.path.join(CAPTURES_DIR, f))
        except Exception:
            pass
        st.rerun()

    # ── Session Log ──
    st.markdown('<div class="ctrl-section">📋 Session Log</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()
    log_placeholder.text("No events yet.")

    # ── Real-time Alerts (LEFT PANEL) ──
    st.markdown('<div class="ctrl-section">🚨 Real-time Alerts</div>', unsafe_allow_html=True)
    alert_placeholder = st.empty()
    alert_placeholder.markdown(
        "<div style='color:#475569; font-size:0.78rem; padding:8px;'>No alerts yet...</div>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close ctrl-panel

# ════════════════════════════════════════════
# RIGHT PANEL — Video Feed + Metrics
# ════════════════════════════════════════════

# Metric placeholders (defined outside column so render_metrics is globally accessible)
with right_col:
    m1, m2, m3, m4 = st.columns(4)
    with m1: suspicious_metric = st.empty()
    with m2: normal_metric     = st.empty()
    with m3: frames_metric     = st.empty()
    with m4: status_metric     = st.empty()

# ── render_metrics defined at module scope so detection loop can call it ──
def render_metrics(sus, norm, frames, is_live):
    suspicious_metric.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#ef4444;">&#128680; {sus}</div>
        <div class="metric-label">Suspicious</div></div>""", unsafe_allow_html=True)
    normal_metric.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#22c55e;">&#9989; {norm}</div>
        <div class="metric-label">Normal</div></div>""", unsafe_allow_html=True)
    frames_metric.markdown(f"""<div class="metric-card">
        <div class="metric-value" style="color:#60a5fa;">&#127902; {frames}</div>
        <div class="metric-label">Frames</div></div>""", unsafe_allow_html=True)
    status_metric.markdown(
        '<div class="metric-card"><div class="status-live">&#128308; LIVE</div>'
        '<div class="metric-label" style="margin-top:6px;">Status</div></div>'
        if is_live else
        '<div class="metric-card"><div class="status-idle">&#9898; IDLE</div>'
        '<div class="metric-label" style="margin-top:6px;">Status</div></div>',
        unsafe_allow_html=True
    )

render_metrics(st.session_state.suspicious_count, st.session_state.normal_count,
               st.session_state.frames_processed, st.session_state.running)

with right_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">&#128247; Live Video Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    video_placeholder.markdown(
        "<div style='background:#0a0e1a; border:1px solid #1e293b; border-radius:12px;"
        "height:420px; display:flex; align-items:center; justify-content:center;"
        "color:#334155; font-size:1rem;'>Press START to begin surveillance</div>",
        unsafe_allow_html=True
    )

# ─── Captures Gallery (full-width below) ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">&#128247; Captured Shoplifters</div>', unsafe_allow_html=True)
gallery_placeholder = st.empty()

def render_gallery(captures, max_n):
    if not captures:
        gallery_placeholder.markdown(
            "<div style='color:#475569; font-size:0.83rem; padding:10px;'>"
            "No suspicious frames captured yet.</div>",
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
                    f'<div class="cap-score">Frame #{cap["frame"]} &nbsp;|&nbsp; Score {cap["score"]:.2f}</div>'
                    f'</div>'
                )
        except Exception as e:
            print(f"Gallery render error: {e}")
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
    st.session_state.alarm_active     = False
    st.session_state.running          = True

    # Reload disk captures if session is empty
    if not st.session_state.captures:
        try:
            files = [os.path.join(CAPTURES_DIR, f) for f in os.listdir(CAPTURES_DIR) if f.endswith('.jpg')]
            files.sort(key=os.path.getmtime, reverse=True)
            for f in files:
                fname = os.path.basename(f)
                try:
                    parts = fname.split('_')
                    ts_str = f"{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:]}"
                    frm = int(parts[3][1:])
                    sc  = float(parts[4][1:-4])
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
        st.error(f"❌ Cannot open source: {cv_source}. Make sure the file exists in the project folder.")
        st.session_state.running = False
        st.stop()

    # ── Preflight: verify the file actually has readable video frames ──
    ret_test, _ = cap.read()
    if not ret_test:
        cap.release()
        st.session_state.running = False
        ext = str(cv_source).split('.')[-1].lower() if isinstance(cv_source, str) else ''
        if ext in ('mp3', 'wav', 'aac', 'ogg', 'm4a'):
            st.error(
                f"❌ **'{cv_source}' is an audio-only file** — it has no video frames.\n\n"
                f"Please rename your file to **tamil.mp4** if it contains video, "
                f"or select a proper video file (mp4, avi, etc.)."
            )
        else:
            st.error(f"❌ Could not read any frames from '{cv_source}'. The file may be corrupt or unsupported.")
        st.stop()
    # Seek back to start after preflight read
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_capture_frame = -60

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

                df_kp    = pd.DataFrame([vals], columns=cols)
                dmatrix  = xgb.DMatrix(df_kp)
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

        # Save capture
        cur_frame = st.session_state.frames_processed
        if frame_has_suspicious and (cur_frame - last_capture_frame) >= 45:
            last_capture_frame = cur_frame
            try:
                path = save_capture(annotated_frame, cur_frame, best_sus_score)
                st.session_state.captures.insert(0, {
                    "path": path, "time": datetime.now().strftime("%H:%M:%S"),
                    "frame": cur_frame, "score": best_sus_score
                })
                render_gallery(st.session_state.captures, max_captures)
            except Exception as e:
                print(f"Capture error: {e}")

        # Alarm control
        if sound_enabled:
            if frame_has_suspicious and not st.session_state.alarm_active:
                set_alarm('play')
                st.session_state.alarm_active = True
            elif not frame_has_suspicious and st.session_state.alarm_active:
                set_alarm('stop')
                st.session_state.alarm_active = False

        st.session_state.frames_processed += 1

        # Update video feed
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update metrics
        render_metrics(st.session_state.suspicious_count, st.session_state.normal_count,
                       st.session_state.frames_processed, True)

        # Update alerts panel (left panel)
        alerts_html = '<div class="alert-scroll">'
        alerts_html += "".join(
            f'<div class="alert-item"><strong>&#128680; {a["time"]}</strong><br>'
            f'Suspicious detected<br>'
            f'<span style="color:#f87171; font-size:0.7rem;">Frame #{a["frame"]} &bull; Score: {a["conf"]:.3f}</span></div>'
            for a in st.session_state.alerts[:20]
        ) or "<div style='color:#475569; font-size:0.78rem; padding:8px;'>No alerts yet...</div>"
        alerts_html += '</div>'
        alert_placeholder.markdown(alerts_html, unsafe_allow_html=True)

        # Update session log
        log_lines = "\n".join(f"[{a['time']}] Frame#{a['frame']}" for a in st.session_state.alerts[:8])
        log_placeholder.text(log_lines or "No events yet.")

    cap.release()
    set_alarm('stop')
    st.session_state.alarm_active = False
    st.session_state.running = False

    render_metrics(st.session_state.suspicious_count, st.session_state.normal_count,
                   st.session_state.frames_processed, False)
    render_gallery(st.session_state.captures, max_captures)
    st.success(f"✅ Detection stopped. {len(st.session_state.captures)} suspicious frames captured.")
