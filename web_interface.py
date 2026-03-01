"""
Face Recognition — Simple Streamlit App
Run with:  streamlit run web_interface.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from face_recognition_system import TrainableFaceRecognizer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition",
    page_icon="🎭",
    layout="centered",
)

st.markdown("""
<style>
    .block-container { max-width: 780px; padding-top: 2rem; }
    h1 { font-size: 1.7rem !important; }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────────
DATA_DIR        = Path("webcam_training_data")
SAMPLES_DIR     = DATA_DIR / "persons"
TARGET_SAMPLES  = 30   # photos per person

# ── Session state defaults ──────────────────────────────────────────────────────
def _init():
    defaults = {
        "page":            "home",   # home | capture | train | recognize
        "person_name":     "",
        "captured_count":  0,
        "model_trained":   False,
        "recognizer":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Helpers ─────────────────────────────────────────────────────────────────────
def get_recognizer() -> TrainableFaceRecognizer:
    if st.session_state.recognizer is None:
        st.session_state.recognizer = TrainableFaceRecognizer(
            model_type="lbph",
            data_dir=str(DATA_DIR),
            unknown_threshold=55.0,
        )
        if st.session_state.recognizer.is_trained:
            st.session_state.model_trained = True
    return st.session_state.recognizer


def detect_faces(gray: np.ndarray, recognizer: TrainableFaceRecognizer):
    return recognizer.face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )


def annotate_frame(frame: np.ndarray, recognizer: TrainableFaceRecognizer) -> np.ndarray:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, recognizer)
    for (x, y, w, h) in faces:
        name, conf = recognizer.predict(gray[y:y+h, x:x+w])
        color = (0, 200, 80) if name != "Unknown" else (30, 60, 220)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-26), (x+w, y), color, -1)
        cv2.putText(frame, f"{name}  {conf:.0f}%",
                    (x+4, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return frame


def save_face_crop(frame_bgr: np.ndarray, person_dir: Path) -> bool:
    """Detect first face, save grayscale crop. Returns True on success."""
    recognizer = get_recognizer()
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, recognizer)
    if len(faces) == 0:
        return False
    x, y, w, h = faces[0]
    crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    idx  = len(list(person_dir.glob("*.jpg")))
    cv2.imwrite(str(person_dir / f"{idx:04d}.jpg"), crop)
    return True


def pil_to_bgr(pil_img) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.title("🎭 Face Recognition")
    st.caption("Capture your face from webcam · Train the model · Recognize in real time")
    st.markdown("---")

    recognizer = get_recognizer()

    persons    = [p.name for p in SAMPLES_DIR.iterdir() if p.is_dir()] if SAMPLES_DIR.exists() else []
    total_imgs = sum(len(list((SAMPLES_DIR / p).glob("*.jpg"))) for p in persons) if persons else 0

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("People registered", len(persons))
    col_b.metric("Training images",   total_imgs)
    col_c.metric("Model", "✅ Ready" if recognizer.is_trained else "⚠️ Not trained")

    st.markdown("---")
    st.subheader("What would you like to do?")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📷  Capture faces", use_container_width=True):
            st.session_state.page = "capture"
            st.rerun()
    with c2:
        label = "🔄  Re-train" if recognizer.is_trained else "🧠  Train model"
        if st.button(label, use_container_width=True, disabled=(len(persons) == 0)):
            st.session_state.page = "train"
            st.rerun()
    with c3:
        if st.button("🎥  Live recognize", use_container_width=True,
                     disabled=(not recognizer.is_trained)):
            st.session_state.page = "recognize"
            st.rerun()

    if persons:
        st.markdown("---")
        st.subheader("Registered people")
        cols = st.columns(min(len(persons), 5))
        for i, name in enumerate(sorted(persons)):
            n = len(list((SAMPLES_DIR / name).glob("*.jpg")))
            cols[i % 5].markdown(f"**{name}**\n\n`{n} photos`")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════
def page_capture():
    st.title("📷 Capture Face Samples")

    if st.button("← Back"):
        st.session_state.page = "home"
        st.session_state.captured_count = 0
        st.rerun()

    st.markdown("---")

    # Name input
    name_input = st.text_input(
        "Person's name",
        value=st.session_state.person_name,
        placeholder="e.g. Alice",
    )
    if name_input != st.session_state.person_name:
        st.session_state.person_name     = name_input
        st.session_state.captured_count  = 0

    if not name_input.strip():
        st.info("Enter a name above to get started.")
        return

    name       = name_input.strip().lower().replace(" ", "_")
    person_dir = SAMPLES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    existing   = len(list(person_dir.glob("*.jpg")))

    # Progress
    st.progress(min(existing / TARGET_SAMPLES, 1.0),
                text=f"{existing} / {TARGET_SAMPLES} samples captured")

    if existing >= TARGET_SAMPLES:
        st.success(f"✅  {TARGET_SAMPLES} samples captured for **{name}**!")
        if st.button("Go to Train →"):
            st.session_state.page = "train"
            st.rerun()
        return

    st.markdown("**Look at the camera, then click the shutter button below.**")

    # key changes after each capture so the widget resets
    camera_frame = st.camera_input("Take a photo", key=f"cam_{existing}")

    if camera_frame is not None:
        frame_bgr = pil_to_bgr(Image.open(camera_frame))
        if save_face_crop(frame_bgr, person_dir):
            st.session_state.captured_count = existing + 1
            st.success(f"✅  Saved! ({existing + 1}/{TARGET_SAMPLES})")
            time.sleep(0.25)
            st.rerun()
        else:
            st.warning("⚠️  No face detected. Move closer or improve lighting.")

    with st.expander("Tips for better accuracy"):
        st.markdown("""
- Look straight at the camera for some shots, then tilt slightly left/right/up/down
- Try different lighting conditions if possible
- Avoid sunglasses or heavy shadows across your face
- Aim for at least 20 clear photos
        """)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
def page_train():
    st.title("🧠 Train Model")

    if st.button("← Back"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")

    if not SAMPLES_DIR.exists():
        st.error("No training data found. Capture some faces first.")
        return

    persons = [p.name for p in SAMPLES_DIR.iterdir() if p.is_dir()]
    if not persons:
        st.error("No person folders found. Go capture faces first.")
        return

    # Summary
    st.subheader("Training data")
    total = 0
    for person in sorted(persons):
        n = len(list((SAMPLES_DIR / person).glob("*.jpg")))
        total += n
        icon = "🟢" if n >= 20 else ("🟡" if n >= 10 else "🔴")
        st.markdown(f"{icon} **{person}** — {n} photos")
    st.markdown(f"**Total: {total} images · {len(persons)} people**")

    if any(len(list((SAMPLES_DIR / p).glob("*.jpg"))) < 5 for p in persons):
        st.warning("⚠️  Some people have fewer than 5 photos — accuracy may suffer.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    model_type = col1.selectbox("Model", ["lbph", "eigenfaces", "fisherfaces"])
    val_split  = col2.slider("Validation split", 0.0, 0.4, 0.2, 0.05)
    threshold  = st.slider("Unknown threshold (%)", 0, 100, 55,
                            help="Faces below this confidence score are labelled 'Unknown'")

    if st.button("🚀  Start Training", type="primary"):
        # Fresh recognizer with chosen settings
        st.session_state.recognizer = TrainableFaceRecognizer(
            model_type=model_type,
            data_dir=str(DATA_DIR),
            unknown_threshold=float(threshold),
        )
        recognizer = st.session_state.recognizer

        with st.spinner("Training… this usually takes a few seconds."):
            try:
                meta = recognizer.train_model(
                    dataset_path=str(SAMPLES_DIR),
                    structure_type="person_folders",
                    validation_split=val_split,
                )
                st.session_state.model_trained = True
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                return

        st.success("✅  Model trained successfully!")
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Samples used", meta.get("num_samples", "—"))
        c2.metric("Classes",      meta.get("num_classes",  "—"))
        c3.metric("Train time",   f"{meta.get('training_time', 0):.1f}s")

        st.markdown("---")
        if st.button("▶  Go to Live Recognition"):
            st.session_state.page = "recognize"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
def page_recognize():
    st.title("🎥 Live Recognition")

    if st.button("← Back"):
        st.session_state.page = "home"
        st.rerun()

    recognizer = get_recognizer()
    if not recognizer.is_trained:
        st.error("Model is not trained yet. Please train first.")
        return

    st.caption(
        f"Model: **{recognizer.model_type.upper()}**  ·  "
        f"Classes: **{len(recognizer.label_to_name)}**  ·  "
        f"Threshold: **{recognizer.unknown_threshold:.0f}%**"
    )
    st.markdown("---")

    mode = st.radio("Mode", ["📸 Single snapshot", "🔴 Continuous webcam"],
                    horizontal=True)

    frame_slot  = st.empty()
    result_slot = st.empty()

    # ── Single snapshot ────────────────────────────────────────────────────────
    if mode == "📸 Single snapshot":
        camera_frame = st.camera_input("Take a snapshot", key="recog_snap")

        if camera_frame is not None:
            frame_bgr     = pil_to_bgr(Image.open(camera_frame))
            annotated     = annotate_frame(frame_bgr.copy(), recognizer)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             use_container_width=True)

            gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, recognizer)

            if len(faces) == 0:
                result_slot.info("No faces detected in this snapshot.")
            else:
                import pandas as pd
                rows = []
                for (x, y, w, h) in faces:
                    name, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    rows.append({
                        "Person":     name,
                        "Confidence": f"{conf:.1f}%",
                        "Status":     "✅ Known" if name != "Unknown" else "❓ Unknown",
                    })
                result_slot.dataframe(pd.DataFrame(rows),
                                      hide_index=True, use_container_width=True)

    # ── Continuous webcam ──────────────────────────────────────────────────────
    else:
        st.info("Opens your webcam and recognizes faces continuously. Click **Stop** to end.")
        stop_btn = st.button("⏹  Stop")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam — make sure it is connected and not in use by another app.")
            return

        try:
            import pandas as pd
            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Lost camera feed.")
                    break

                annotated = annotate_frame(frame.copy(), recognizer)
                frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_container_width=True)

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces(gray, recognizer)
                if len(faces) > 0:
                    rows = []
                    for (x, y, w, h) in faces:
                        name, conf = recognizer.predict(gray[y:y+h, x:x+w])
                        rows.append({
                            "Person":     name,
                            "Confidence": f"{conf:.1f}%",
                            "Status":     "✅ Known" if name != "Unknown" else "❓ Unknown",
                        })
                    result_slot.dataframe(pd.DataFrame(rows),
                                          hide_index=True, use_container_width=True)
                else:
                    result_slot.caption("No faces in frame")

                time.sleep(0.04)   # ~25 fps

        finally:
            cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
{
    "home":      page_home,
    "capture":   page_capture,
    "train":     page_train,
    "recognize": page_recognize,
}.get(st.session_state.page, page_home)()
