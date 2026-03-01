
import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from face_recognition_system import TrainableFaceRecognizer


st.set_page_config(page_title="Face Recognition System", layout="wide")

st.title("🧠 Face Recognition System - Web Interface")
st.caption("Upload images, run prediction, and train/evaluate with a simple UI.")

with st.sidebar:
    st.header("Model Settings")
    model_type = st.selectbox("Model type", ["lbph", "eigenfaces", "fisherfaces"], index=0)
    data_dir = st.text_input("Data directory", value="training_data")
    unknown_threshold = st.slider("Unknown threshold (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)

    st.header("Dataset Settings")
    structure = st.selectbox("Dataset structure", ["person_folders", "flat_with_labels", "csv_labeled"], index=0)
    validation_split = st.slider("Validation split", min_value=0.0, max_value=0.5, value=0.2, step=0.05)


@st.cache_resource
def get_recognizer(model_type, data_dir, unknown_threshold):
    return TrainableFaceRecognizer(
        model_type=model_type,
        data_dir=data_dir,
        unknown_threshold=unknown_threshold,
    )


recognizer = get_recognizer(model_type, data_dir, unknown_threshold)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Analyze or Train Dataset")
    dataset_path = st.text_input("Dataset path", value="sample_test_faces")

    if st.button("Analyze Dataset"):
        try:
            report = recognizer.analyze_dataset(dataset_path, structure_type=structure, save_report=True)
            st.success("Dataset analyzed successfully")
            st.json(report)
        except Exception as exc:
            st.error(f"Dataset analysis failed: {exc}")

    if st.button("Train Model"):
        try:
            metadata = recognizer.train_model(dataset_path, structure_type=structure, validation_split=validation_split)
            st.success("Model trained successfully")
            st.json(metadata)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

with col2:
    st.subheader("2) Upload Image for Prediction")
    uploaded_file = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict Faces"):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getbuffer())

                    result = recognizer.predict_image_file(
                        tmp_path,
                        save_annotated=True,
                        output_dir=Path(data_dir) / "web_annotated",
                    )

                st.success(f"Prediction done. Faces found: {result['num_faces']}")
                st.json(result)

                annotated_path = result.get("annotated_image")
                if annotated_path and Path(annotated_path).exists():
                    st.image(str(annotated_path), caption="Annotated prediction", use_container_width=True)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

st.subheader("3) Evaluate Existing Model")
eval_dataset = st.text_input("Evaluation dataset path", value="sample_test_faces")
if st.button("Evaluate"):
    try:
        test_data = recognizer.load_dataset_from_directory(eval_dataset, structure)
        data = recognizer.prepare_training_data(test_data, validation_split=0)
        test_faces, test_labels, _ = data["train"]
        evaluation = recognizer.evaluate_model(test_faces, test_labels, "Web Eval", save_report=True)
        st.success("Evaluation complete")
        st.json(evaluation)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")

st.markdown("---")
st.caption("Tip: run this app with `streamlit run web_interface.py`.")

# Minimal debug info
with st.expander("Debug Info"):
    st.code(json.dumps({
        "model_type": model_type,
        "data_dir": data_dir,
        "unknown_threshold": unknown_threshold,
        "is_trained": recognizer.is_trained,
        "num_classes": len(recognizer.label_to_name),
    }, indent=2))
