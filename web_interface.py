"""
Face Recognition — Simple Streamlit App
Run with:  python -m streamlit run web_interface.py
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

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Recognition", page_icon="🎭", layout="centered")

st.markdown("""
<style>
    .block-container { max-width: 780px; padding-top: 2rem; }
    h1 { font-size: 1.7rem !important; }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("webcam_training_data")
SAMPLES_DIR    = DATA_DIR / "persons"
TARGET_SAMPLES = 50   # more samples = better accuracy

# ── Session state ────────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "page":           "home",
        "person_name":    "",
        "model_trained":  False,
        "recognizer":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Helpers ──────────────────────────────────────────────────────────────────────
def get_recognizer(force_new: bool = False) -> TrainableFaceRecognizer:
    """Return cached recognizer. Pass force_new=True to discard old model."""
    if force_new or st.session_state.recognizer is None:
        st.session_state.recognizer = TrainableFaceRecognizer(
            model_type="lbph",
            data_dir=str(DATA_DIR),
            unknown_threshold=55.0,
        )
        st.session_state.model_trained = st.session_state.recognizer.is_trained
    return st.session_state.recognizer


def delete_old_model():
    """Wipe saved model files so training always starts fresh."""
    for fname in ["lbph_model.yml", "eigenfaces_model.yml", "fisherfaces_model.yml",
                  "labels.pkl", "training_metadata.json"]:
        f = DATA_DIR / fname
        if f.exists():
            f.unlink()
    # Also clear cached recognizer
    st.session_state.recognizer  = None
    st.session_state.model_trained = False


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
        cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
        cv2.putText(frame, f"{name}  {conf:.0f}%",
                    (x+4, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def augment_and_save(gray_crop: np.ndarray, person_dir: Path):
    """
    Save the original crop + several augmented versions.
    More variety = better accuracy without needing 100s of manual photos.
    """
    base_idx = len(list(person_dir.glob("*.jpg")))
    crops = [gray_crop]

    # Horizontal flip
    crops.append(cv2.flip(gray_crop, 1))

    # Brightness variations
    for alpha in (0.75, 1.25):
        crops.append(np.clip(gray_crop * alpha, 0, 255).astype(np.uint8))

    # Slight blur (simulates out-of-focus)
    crops.append(cv2.GaussianBlur(gray_crop, (3, 3), 0))

    # Small rotation (+8° and -8°)
    h, w = gray_crop.shape
    cx, cy = w // 2, h // 2
    for angle in (8, -8):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        crops.append(cv2.warpAffine(gray_crop, M, (w, h)))

    for i, crop in enumerate(crops):
        cv2.imwrite(str(person_dir / f"{base_idx + i:04d}.jpg"), crop)

    return len(crops)  # number of images saved


def save_face_crop(frame_bgr: np.ndarray, person_dir: Path):
    """
    Detect face in frame, apply augmentation, save all crops.
    Returns (success: bool, n_saved: int).
    """
    recognizer = get_recognizer()
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, recognizer)
    if len(faces) == 0:
        return False, 0
    x, y, w, h = faces[0]
    crop   = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    n_saved = augment_and_save(crop, person_dir)
    return True, n_saved


def pil_to_bgr(pil_img) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def person_count() -> list:
    if not SAMPLES_DIR.exists():
        return []
    return [p.name for p in sorted(SAMPLES_DIR.iterdir()) if p.is_dir()]


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.title("🎭 Face Recognition")
    st.caption("Capture · Train · Recognize")
    st.markdown("---")

    recognizer = get_recognizer()
    persons    = person_count()
    total_imgs = sum(len(list((SAMPLES_DIR / p).glob("*.jpg"))) for p in persons) if persons else 0

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("People", len(persons))
    col_b.metric("Training images", total_imgs)
    col_c.metric("Model", "✅ Ready" if recognizer.is_trained else "⚠️ Not trained")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📷  Capture faces", use_container_width=True):
            st.session_state.page = "capture"
            st.rerun()
    with c2:
        label = "🔄  Re-train" if recognizer.is_trained else "🧠  Train model"
        if st.button(label, use_container_width=True, disabled=len(persons) == 0):
            st.session_state.page = "train"
            st.rerun()
    with c3:
        if st.button("🎥  Recognize", use_container_width=True,
                     disabled=not recognizer.is_trained):
            st.session_state.page = "recognize"
            st.rerun()

    # Registered people with manage option
    if persons:
        st.markdown("---")
        st.subheader("Registered people")
        for name in persons:
            n = len(list((SAMPLES_DIR / name).glob("*.jpg")))
            col_n, col_d = st.columns([4, 1])
            icon = "🟢" if n >= 40 else ("🟡" if n >= 20 else "🔴")
            col_n.markdown(f"{icon} **{name}** — {n} images")
            if col_d.button("🗑 Delete", key=f"del_{name}"):
                shutil.rmtree(SAMPLES_DIR / name)
                st.success(f"Deleted {name}. Re-train for changes to take effect.")
                delete_old_model()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════
def page_capture():
    st.title("📷 Capture Face Samples")

    if st.button("← Back"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")
    st.info(f"Each photo you take is saved as **7 augmented versions** automatically, "
            f"so {TARGET_SAMPLES // 7} photos ≈ {TARGET_SAMPLES}+ training images.")

    name_input = st.text_input(
        "Person's name",
        value=st.session_state.person_name,
        placeholder="e.g. Alice",
    )
    if name_input != st.session_state.person_name:
        st.session_state.person_name = name_input

    if not name_input.strip():
        st.info("Enter a name above to get started.")
        return

    name       = name_input.strip().lower().replace(" ", "_")
    person_dir = SAMPLES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    existing   = len(list(person_dir.glob("*.jpg")))

    # ── Progress ─────────────────────────────────────────────────────────────
    progress = min(existing / TARGET_SAMPLES, 1.0)
    st.progress(progress, text=f"{existing} / {TARGET_SAMPLES} training images")

    # Option to wipe and restart this person
    col_cam, col_reset = st.columns([3, 1])
    with col_reset:
        if st.button("🗑 Clear & restart", help="Delete all photos for this person and start over"):
            shutil.rmtree(person_dir)
            person_dir.mkdir(parents=True, exist_ok=True)
            delete_old_model()
            st.success("Cleared. Start capturing again.")
            st.rerun()

    if existing >= TARGET_SAMPLES:
        st.success(f"✅  Enough samples for **{name}**! You can now train.")
        if st.button("Go to Train →"):
            st.session_state.page = "train"
            st.rerun()
        return

    with col_cam:
        st.markdown("**Look at the camera and click the shutter. Vary your angle slightly each shot.**")

    camera_frame = st.camera_input("Take a photo", key=f"cam_{existing}")

    if camera_frame is not None:
        frame_bgr = pil_to_bgr(Image.open(camera_frame))
        found, n_saved = save_face_crop(frame_bgr, person_dir)
        if found:
            new_total = existing + n_saved
            st.success(f"✅  Saved {n_saved} images (1 photo + augmentations) — total: {new_total}")
            time.sleep(0.3)
            st.rerun()
        else:
            st.warning("⚠️  No face detected. Move closer, face the camera, and try again.")

    with st.expander("Tips for best accuracy"):
        st.markdown("""
- **Vary your angle** — straight on, slightly left, slightly right, chin up/down
- **Vary lighting** — try near a window, then away from it
- **Different expressions** — neutral, slight smile, serious
- **No sunglasses or heavy shadows** across your eyes/face
- Aim for **at least 8–10 good photos** (= 56–70 training images after augmentation)
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

    persons = person_count()
    if not persons:
        st.error("No training data found. Capture some faces first.")
        return

    # Data summary
    st.subheader("Training data")
    total = 0
    low_count = False
    for person in persons:
        n = len(list((SAMPLES_DIR / person).glob("*.jpg")))
        total += n
        icon = "🟢" if n >= 40 else ("🟡" if n >= 20 else "🔴")
        if n < 20:
            low_count = True
        st.markdown(f"{icon} **{person}** — {n} images")

    st.markdown(f"**Total: {total} images across {len(persons)} people**")

    if low_count:
        st.warning("⚠️  Some people have fewer than 20 images. Capture more photos for better accuracy.")

    st.markdown("---")

    # Settings
    col1, col2 = st.columns(2)
    model_type = col1.selectbox("Model type", ["lbph", "eigenfaces", "fisherfaces"],
                                 help="LBPH works best for small datasets")
    val_split  = col2.slider("Validation split", 0.0, 0.4, 0.2, 0.05)
    threshold  = st.slider("Unknown threshold (%)", 0, 100, 55,
                            help="Predictions below this % are shown as Unknown")

    st.markdown("---")

    # Always wipe old model before training so it truly retrains from scratch
    st.caption("ℹ️  Clicking Train will **delete the old model** and build a fresh one from your current photos.")

    if st.button("🚀  Start Training", type="primary"):

        # 1. Delete old saved model files
        delete_old_model()

        # 2. Create fresh recognizer (no model loaded from disk)
        recognizer = TrainableFaceRecognizer(
            model_type=model_type,
            data_dir=str(DATA_DIR),
            unknown_threshold=float(threshold),
        )
        st.session_state.recognizer = recognizer

        # 3. Train
        with st.spinner("Training… please wait."):
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

        st.success("✅  Model trained and saved!")
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
        f"People: **{len(recognizer.label_to_name)}**  ·  "
        f"Threshold: **{recognizer.unknown_threshold:.0f}%**"
    )
    st.markdown("---")

    mode = st.radio("Mode", ["📸 Single snapshot", "🔴 Continuous webcam"], horizontal=True)

    frame_slot  = st.empty()
    result_slot = st.empty()

    # ── Single snapshot ────────────────────────────────────────────────────────
    if mode == "📸 Single snapshot":
        camera_frame = st.camera_input("Take a snapshot", key="recog_snap")

        if camera_frame is not None:
            frame_bgr = pil_to_bgr(Image.open(camera_frame))
            annotated = annotate_frame(frame_bgr.copy(), recognizer)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             use_container_width=True)

            gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, recognizer)

            if len(faces) == 0:
                result_slot.info("No faces detected.")
            else:
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
        st.info("Streams your webcam with live face recognition. Click **Stop** to end.")
        stop_btn = st.button("⏹  Stop")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Make sure it is connected and not used by another app.")
            return

        try:
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

                time.sleep(0.04)
        finally:
            cap.release()


# ── Router ───────────────────────────────────────────────────────────────────────
{
    "home":      page_home,
    "capture":   page_capture,
    "train":     page_train,
    "recognize": page_recognize,
}.get(st.session_state.page, page_home)()
