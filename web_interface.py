"""
NEXUS — Multi-Person Face Recognition System
Run:  python -m streamlit run web_interface.py
"""

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from face_recognition_system import TrainableFaceRecognizer

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS · Face Recognition",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #080c10;
    --surface:  #0d1117;
    --border:   #1c2433;
    --accent:   #00e5ff;
    --green:    #00ff88;
    --red:      #ff3b5c;
    --amber:    #ffb300;
    --text:     #cdd9e5;
    --muted:    #4a6070;
    --font-ui:  'JetBrains Mono', monospace;
    --font-hd:  'Syne', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-ui) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main area ── */
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-ui) !important; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: var(--font-hd) !important;
    letter-spacing: -0.02em;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-hd) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    transition: all 0.15s ease !important;
    padding: 0.45rem 1rem !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: rgba(0,229,255,0.08) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover {
    background: rgba(0,229,255,0.18) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.25) !important;
}

/* ── Inputs / selects ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stSlider > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.82rem !important;
    border-radius: 3px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px rgba(0,229,255,0.2) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 8px var(--accent) !important;
}

/* ── Radio ── */
.stRadio label { font-size: 0.8rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--green)) !important;
}

/* ── Camera input ── */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border-radius: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stCameraInput"] button {
    background: var(--accent) !important;
    color: #000 !important;
    border-radius: 50% !important;
    font-weight: 700 !important;
}

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 3px !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--surface) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ── Custom components ── */
.nexus-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.1rem;
}
.nexus-logo {
    font-family: var(--font-hd);
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent);
    letter-spacing: -0.03em;
}
.nexus-sub {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--muted);
    text-transform: uppercase;
}
.person-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.15s;
}
.person-card:hover { border-color: var(--accent); }
.person-name {
    font-family: var(--font-hd);
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text);
}
.person-meta {
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
}
.dot-green  { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
.dot-amber  { background: #ffb300; box-shadow: 0 0 6px #ffb300; }
.dot-red    { background: #ff3b5c; box-shadow: 0 0 6px #ff3b5c; }
.tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 2px;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    font-weight: 600;
    text-transform: uppercase;
}
.tag-green { background: rgba(0,255,136,0.1); color: #00ff88; border: 1px solid rgba(0,255,136,0.25); }
.tag-amber { background: rgba(255,179,0,0.1);  color: #ffb300; border: 1px solid rgba(255,179,0,0.25); }
.tag-red   { background: rgba(255,59,92,0.1);  color: #ff3b5c; border: 1px solid rgba(255,59,92,0.25); }
.tag-blue  { background: rgba(0,229,255,0.1);  color: #00e5ff; border: 1px solid rgba(0,229,255,0.25); }
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 12px;
    margin-top: 4px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.75rem;
}
.stat-key { color: var(--muted); letter-spacing: 0.05em; }
.stat-val { color: var(--text); font-weight: 600; }
.stat-val.accent { color: var(--accent); }
.stat-val.green  { color: var(--green); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("nexus_data")
SAMPLES_DIR   = DATA_DIR / "persons"
TARGET_IMGS   = 50   # target training images per person (after augmentation)
AUGMENT_FACTOR = 7  # augmentations per captured photo

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "page":          "dashboard",
        "capture_name":  "",
        "recognizer":    None,
        "model_trained": False,
        "train_meta":    None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────────────────────
#  CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_recognizer(force_new: bool = False) -> TrainableFaceRecognizer:
    if force_new or st.session_state.recognizer is None:
        st.session_state.recognizer = TrainableFaceRecognizer(
            model_type="lbph",
            data_dir=str(DATA_DIR),
            unknown_threshold=55.0,
        )
        st.session_state.model_trained = st.session_state.recognizer.is_trained
    return st.session_state.recognizer


def wipe_model():
    for fname in ["lbph_model.yml", "eigenfaces_model.yml",
                  "fisherfaces_model.yml", "labels.pkl", "training_metadata.json"]:
        f = DATA_DIR / fname
        if f.exists():
            f.unlink()
    st.session_state.recognizer    = None
    st.session_state.model_trained = False
    st.session_state.train_meta    = None


def persons_list() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if p.is_dir())


def person_img_count(name: str) -> int:
    d = SAMPLES_DIR / name
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def person_health(n: int) -> tuple[str, str]:
    """Returns (dot_class, tag_class, label)."""
    if n >= 40:   return "dot-green", "tag-green", "GOOD"
    if n >= 20:   return "dot-amber", "tag-amber", "LOW"
    return "dot-red", "tag-red", "POOR"


def pil_to_bgr(img) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_faces(gray, recognizer):
    return recognizer.face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
    )


def augment_and_save(gray_crop: np.ndarray, person_dir: Path) -> int:
    base = len(list(person_dir.glob("*.jpg")))
    crops = [gray_crop, cv2.flip(gray_crop, 1)]
    for alpha in (0.7, 1.3):
        crops.append(np.clip(gray_crop * alpha, 0, 255).astype(np.uint8))
    crops.append(cv2.GaussianBlur(gray_crop, (3, 3), 0))
    h, w = gray_crop.shape[:2]
    cx, cy = w // 2, h // 2
    for angle in (10, -10):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        crops.append(cv2.warpAffine(gray_crop, M, (w, h)))
    for i, c in enumerate(crops):
        cv2.imwrite(str(person_dir / f"{base+i:05d}.jpg"), c)
    return len(crops)


def save_face_from_frame(frame_bgr: np.ndarray, person_dir: Path):
    recognizer = get_recognizer()
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, recognizer)
    if len(faces) == 0:
        return False, 0
    x, y, w, h = faces[0]
    crop   = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    n      = augment_and_save(crop, person_dir)
    return True, n


def annotate_bgr(frame: np.ndarray, recognizer: TrainableFaceRecognizer) -> tuple:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, recognizer)
    results = []
    for (x, y, w, h) in faces:
        name, conf = recognizer.predict(gray[y:y+h, x:x+w])
        known  = name != "Unknown"
        color  = (0, 220, 120) if known else (255, 60, 80)
        # Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Corner accents
        L = 14
        for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame, (px, py), (px+dx*L, py), color, 3)
            cv2.line(frame, (px, py), (px, py+dy*L), color, 3)
        # Label bar
        label  = f"  {name}  {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+4, y), color, -1)
        cv2.putText(frame, label, (x+2, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1, cv2.LINE_AA)
        results.append({"name": name, "confidence": conf, "known": known})
    return frame, results

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="nexus-logo">⬡ NEXUS</div>', unsafe_allow_html=True)
        st.markdown('<div class="nexus-sub">Face Recognition System</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        pages = {
            "dashboard": "⬡  Dashboard",
            "capture":   "◎  Capture",
            "train":     "◈  Train",
            "recognize": "▶  Recognize",
        }
        for key, label in pages.items():
            active = st.session_state.page == key
            style  = "primary" if active else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)

        recognizer = get_recognizer()
        persons    = persons_list()
        total_imgs = sum(person_img_count(p) for p in persons)

        st.markdown(f"""
        <div class="stat-row"><span class="stat-key">STATUS</span>
        <span class="stat-val green">{'ONLINE' if recognizer.is_trained else 'IDLE'}</span></div>
        <div class="stat-row"><span class="stat-key">PERSONS</span>
        <span class="stat-val accent">{len(persons)}</span></div>
        <div class="stat-row"><span class="stat-key">IMAGES</span>
        <span class="stat-val">{total_imgs}</span></div>
        <div class="stat-row"><span class="stat-key">MODEL</span>
        <span class="stat-val">{'LBPH' if recognizer.is_trained else '—'}</span></div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if persons:
            st.markdown('<div class="section-label">Registered</div>', unsafe_allow_html=True)
            for name in persons:
                n = person_img_count(name)
                dot, _, _ = person_health(n)
                st.markdown(
                    f'<div style="font-size:0.75rem;padding:3px 0;color:var(--text);">'
                    f'<span class="status-dot {dot}"></span>{name}'
                    f'<span style="float:right;color:var(--muted);font-size:0.65rem;">{n}</span></div>',
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown('<div class="nexus-header"><span class="nexus-logo">⬡ NEXUS</span>'
                '<span class="nexus-sub">Multi-Person Face Recognition</span></div>',
                unsafe_allow_html=True)
    st.markdown("---")

    recognizer = get_recognizer()
    persons    = persons_list()
    total_imgs = sum(person_img_count(p) for p in persons)
    ready_pct  = (sum(1 for p in persons if person_img_count(p) >= 40) / len(persons) * 100) if persons else 0

    # ── Metrics row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Persons Registered", len(persons))
    c2.metric("Training Images",    total_imgs)
    c3.metric("Model Status",       "READY" if recognizer.is_trained else "UNTRAINED")
    c4.metric("Data Quality",       f"{ready_pct:.0f}%")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-label">Registered Persons</div>', unsafe_allow_html=True)

        if not persons:
            st.info("No persons registered yet. Go to **◎ Capture** to add people.")
        else:
            for name in persons:
                n = person_img_count(name)
                dot, tag_cls, tag_lbl = person_health(n)
                needed = max(0, TARGET_IMGS - n)
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.markdown(
                        f'<div style="padding:8px 0;">'
                        f'<span class="status-dot {dot}"></span>'
                        f'<span style="font-family:\'Syne\',sans-serif;font-weight:700;">{name}</span>'
                        f'&nbsp;&nbsp;<span class="tag {tag_cls}">{tag_lbl}</span>'
                        f'<div style="font-size:0.68rem;color:var(--muted);margin-top:2px;padding-left:12px;">'
                        f'{n} images{f" · needs {needed} more" if needed > 0 else " · ✓ sufficient"}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                with col_b:
                    if st.button("＋ Capture", key=f"cap_{name}", use_container_width=True):
                        st.session_state.capture_name = name
                        st.session_state.page = "capture"
                        st.rerun()
                with col_c:
                    if st.button("🗑", key=f"del_{name}", use_container_width=True):
                        shutil.rmtree(SAMPLES_DIR / name)
                        wipe_model()
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("＋  Register New Person", type="primary", use_container_width=True):
            st.session_state.capture_name = ""
            st.session_state.page = "capture"
            st.rerun()

    with col_right:
        st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
        if st.button("◈  Train / Re-train Model", use_container_width=True,
                     disabled=len(persons) == 0):
            st.session_state.page = "train"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Start Recognition", use_container_width=True, type="primary",
                     disabled=not recognizer.is_trained):
            st.session_state.page = "recognize"
            st.rerun()

        if recognizer.is_trained and st.session_state.train_meta:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Last Training</div>', unsafe_allow_html=True)
            meta = st.session_state.train_meta
            st.markdown(f"""
            <div class="stat-row"><span class="stat-key">SAMPLES</span>
            <span class="stat-val accent">{meta.get('num_samples','—')}</span></div>
            <div class="stat-row"><span class="stat-key">CLASSES</span>
            <span class="stat-val accent">{meta.get('num_classes','—')}</span></div>
            <div class="stat-row"><span class="stat-key">DURATION</span>
            <span class="stat-val">{meta.get('training_time',0):.1f}s</span></div>
            <div class="stat-row"><span class="stat-key">TIMESTAMP</span>
            <span class="stat-val" style="font-size:0.65rem;">{meta.get('timestamp','—')}</span></div>
            """, unsafe_allow_html=True)

        if recognizer.is_trained:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
            classes = list(recognizer.label_to_name.values())
            for cls in classes:
                st.markdown(
                    f'<div style="font-size:0.75rem;padding:2px 0;">'
                    f'<span class="tag tag-blue">{cls}</span></div>',
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: CAPTURE
# ─────────────────────────────────────────────────────────────────────────────
def page_capture():
    st.markdown('<div class="nexus-logo" style="font-size:1.3rem;">◎ CAPTURE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="nexus-sub" style="margin-bottom:1rem;">Register a person · Collect training images</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_tips = st.columns([3, 2])

    with col_form:
        existing_persons = persons_list()

        mode = st.radio("", ["New person", "Add photos to existing"],
                        horizontal=True, label_visibility="collapsed")

        if mode == "New person":
            name_input = st.text_input("Full name", value=st.session_state.capture_name,
                                       placeholder="e.g. John Smith")
        else:
            if not existing_persons:
                st.warning("No persons registered yet. Use 'New person' first.")
                return
            name_input = st.selectbox("Select person", existing_persons)

        if not name_input or not name_input.strip():
            st.info("Enter or select a name to start capturing.")
            return

        name       = name_input.strip().lower().replace(" ", "_")
        person_dir = SAMPLES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        existing   = person_img_count(name)

        # Stats row
        dot, tag_cls, tag_lbl = person_health(existing)
        shots_done = existing // AUGMENT_FACTOR
        shots_need = max(0, (TARGET_IMGS - existing + AUGMENT_FACTOR - 1) // AUGMENT_FACTOR)

        st.markdown(f"""
        <div style="display:flex;gap:16px;padding:10px 0;border-bottom:1px solid var(--border);
                    border-top:1px solid var(--border);margin:12px 0;">
          <div><div class="nexus-sub">IMAGES</div>
               <div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;
                           font-weight:800;color:var(--accent);">{existing}</div></div>
          <div><div class="nexus-sub">PHOTOS TAKEN</div>
               <div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:800;">{shots_done}</div></div>
          <div><div class="nexus-sub">QUALITY</div>
               <div style="padding-top:4px;"><span class="tag {tag_cls}">{tag_lbl}</span></div></div>
          <div><div class="nexus-sub">STILL NEEDED</div>
               <div style="font-family:\'Syne\',sans-serif;font-size:1.4rem;
                           font-weight:800;color:{'var(--green)' if shots_need==0 else 'var(--amber)'};">{shots_need}</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(existing / TARGET_IMGS, 1.0))

        if existing >= TARGET_IMGS:
            st.success(f"✅  Sufficient images for **{name_input}**. Model is ready to train.")
            if st.button("Go to Train →", type="primary"):
                st.session_state.page = "train"
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

        camera_frame = st.camera_input("", key=f"cam_{name}_{existing}",
                                       label_visibility="collapsed")

        col_r, col_c = st.columns(2)
        with col_r:
            if st.button("Clear all photos for this person"):
                shutil.rmtree(person_dir)
                person_dir.mkdir(parents=True, exist_ok=True)
                wipe_model()
                st.rerun()

        if camera_frame is not None:
            frame_bgr = pil_to_bgr(Image.open(camera_frame))
            found, n_saved = save_face_from_frame(frame_bgr, person_dir)
            if found:
                new_total = existing + n_saved
                st.success(f"✅  Saved {n_saved} images (1 photo → {AUGMENT_FACTOR} augmentations) · Total: {new_total}")
                time.sleep(0.25)
                st.rerun()
            else:
                st.warning("⚠️  No face detected — move closer and ensure good lighting.")

    with col_tips:
        st.markdown('<div class="section-label">Tips for High Accuracy</div>', unsafe_allow_html=True)
        tips = [
            ("Vary angles", "Tilt head slightly left, right, up, down between shots"),
            ("Vary lighting", "Shoot near a window, then away — different shadows help"),
            ("Expressions", "Neutral, slight smile, serious — mix them up"),
            ("Distance", "Fill ~60–70% of the frame with your face"),
            ("Minimum shots", f"Take at least {TARGET_IMGS // AUGMENT_FACTOR} photos per person"),
            ("More people", "The system handles unlimited people — just keep adding"),
        ]
        for title, desc in tips:
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<div style="font-size:0.75rem;font-weight:600;color:var(--accent);">{title}</div>'
                f'<div style="font-size:0.72rem;color:var(--muted);line-height:1.5;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-label" style="margin-top:16px;">Augmentation</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;color:var(--muted);line-height:1.8;">'
            f'Each captured photo is auto-expanded into <span style="color:var(--accent);'
            f'font-weight:600;">{AUGMENT_FACTOR} training images</span>:<br>'
            f'Original · Flipped · Brighter · Darker · Blurred · Rotated +10° · Rotated -10°'
            f'</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def page_train():
    st.markdown('<div class="nexus-logo" style="font-size:1.3rem;">◈ TRAIN</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="nexus-sub" style="margin-bottom:1rem;">Build the recognition model from captured data</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    persons = persons_list()
    if not persons:
        st.error("No training data found. Go to Capture first.")
        return

    col_config, col_summary = st.columns([2, 3])

    with col_config:
        st.markdown('<div class="section-label">Model Settings</div>', unsafe_allow_html=True)
        model_type = st.selectbox("Algorithm", ["lbph", "eigenfaces", "fisherfaces"],
                                   help="LBPH is fastest and works best for small datasets")
        val_split  = st.slider("Validation split", 0.0, 0.4, 0.2, 0.05,
                                help="Fraction of data held out for accuracy check")
        threshold  = st.slider("Unknown threshold %", 0, 100, 55,
                                help="Predictions below this are labelled Unknown")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.68rem;color:var(--muted);padding:8px;'
            'border:1px solid var(--border);border-radius:3px;">'
            '⚠ Training will delete the old model and rebuild from all current photos.'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("◈  Start Training", type="primary", use_container_width=True):
            wipe_model()
            recognizer = TrainableFaceRecognizer(
                model_type=model_type,
                data_dir=str(DATA_DIR),
                unknown_threshold=float(threshold),
            )
            st.session_state.recognizer = recognizer

            prog_bar = st.progress(0, text="Preparing data...")
            try:
                prog_bar.progress(20, text="Loading images...")
                meta = recognizer.train_model(
                    dataset_path=str(SAMPLES_DIR),
                    structure_type="person_folders",
                    validation_split=val_split,
                )
                prog_bar.progress(100, text="Done!")
                st.session_state.model_trained = True
                st.session_state.train_meta    = meta
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

            st.success(
                f"✅  Model trained on **{meta.get('num_samples')} images** "
                f"across **{meta.get('num_classes')} people** "
                f"in **{meta.get('training_time',0):.1f}s**"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶  Go to Recognition", type="primary", use_container_width=True):
                st.session_state.page = "recognize"
                st.rerun()

    with col_summary:
        st.markdown('<div class="section-label">Data Summary</div>', unsafe_allow_html=True)
        total = 0
        all_ok = True
        for person in persons:
            n   = person_img_count(person)
            total += n
            dot, tag_cls, tag_lbl = person_health(n)
            if n < 20:
                all_ok = False
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:8px 10px;margin-bottom:4px;background:var(--surface);'
                f'border:1px solid var(--border);border-radius:3px;">'
                f'<span><span class="status-dot {dot}"></span>'
                f'<span style="font-weight:600;">{person}</span></span>'
                f'<span style="display:flex;align-items:center;gap:8px;">'
                f'<span style="color:var(--muted);font-size:0.72rem;">{n} images</span>'
                f'<span class="tag {tag_cls}">{tag_lbl}</span>'
                f'</span></div>',
                unsafe_allow_html=True
            )
        st.markdown(
            f'<div style="margin-top:8px;padding:8px 10px;border-top:1px solid var(--border);">'
            f'<span style="font-size:0.72rem;color:var(--muted);">TOTAL</span>'
            f'&nbsp;&nbsp;<span style="color:var(--accent);font-weight:700;">{total} images</span>'
            f'&nbsp;·&nbsp;<span style="color:var(--text);">{len(persons)} persons</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if not all_ok:
            st.warning("Some persons have fewer than 20 images. Capture more for better accuracy.")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: RECOGNIZE
# ─────────────────────────────────────────────────────────────────────────────
def page_recognize():
    st.markdown('<div class="nexus-logo" style="font-size:1.3rem;">▶ RECOGNIZE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="nexus-sub" style="margin-bottom:1rem;">Identify faces in real time</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    recognizer = get_recognizer()
    if not recognizer.is_trained:
        st.error("Model not trained yet. Go to ◈ Train first.")
        return

    known = list(recognizer.label_to_name.values())
    st.markdown(
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">'
        + "".join(f'<span class="tag tag-blue">{k}</span>' for k in known)
        + f'<span style="font-size:0.68rem;color:var(--muted);align-self:center;">'
          f'  {len(known)} registered · threshold {recognizer.unknown_threshold:.0f}%</span>'
        + '</div>',
        unsafe_allow_html=True
    )

    mode = st.radio("", ["📸 Snapshot", "🔴 Live stream"], horizontal=True,
                    label_visibility="collapsed")

    frame_slot  = st.empty()
    result_slot = st.empty()

    # ── Snapshot ──────────────────────────────────────────────────────────────
    if mode == "📸 Snapshot":
        cam = st.camera_input("", key="recog_snap", label_visibility="collapsed")
        if cam is not None:
            frame_bgr          = pil_to_bgr(Image.open(cam))
            annotated, results = annotate_bgr(frame_bgr.copy(), recognizer)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             use_container_width=True)

            if not results:
                result_slot.info("No faces detected.")
            else:
                rows = [{
                    "Person":     r["name"],
                    "Confidence": f"{r['confidence']:.1f}%",
                    "Status":     "✅ IDENTIFIED" if r["known"] else "❓ UNKNOWN",
                } for r in results]
                result_slot.dataframe(pd.DataFrame(rows),
                                      hide_index=True, use_container_width=True)

    # ── Live stream ───────────────────────────────────────────────────────────
    else:
        col_stop, col_info = st.columns([1, 3])
        stop = col_stop.button("⏹  Stop", type="primary")
        col_info.markdown(
            '<span style="font-size:0.72rem;color:var(--muted);">'
            'Streaming from camera 0 · Press Stop to end</span>',
            unsafe_allow_html=True
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Check it is connected and not in use.")
            return

        frame_count = 0
        try:
            while not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # Run detection every 2nd frame for speed
                if frame_count % 2 == 0:
                    annotated, results = annotate_bgr(frame.copy(), recognizer)
                    frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                     channels="RGB", use_container_width=True)
                    if results:
                        rows = [{
                            "Person":     r["name"],
                            "Confidence": f"{r['confidence']:.1f}%",
                            "Status":     "✅ IDENTIFIED" if r["known"] else "❓ UNKNOWN",
                        } for r in results]
                        result_slot.dataframe(pd.DataFrame(rows),
                                              hide_index=True, use_container_width=True)
                    else:
                        result_slot.markdown(
                            '<span style="font-size:0.72rem;color:var(--muted);">No faces in frame</span>',
                            unsafe_allow_html=True
                        )
                time.sleep(0.03)
        finally:
            cap.release()

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────────────────────
render_sidebar()

{
    "dashboard": page_dashboard,
    "capture":   page_capture,
    "train":     page_train,
    "recognize": page_recognize,
}.get(st.session_state.page, page_dashboard)()
