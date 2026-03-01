# Face Recognition System

A trainable OpenCV face recognition project with practical CLI workflows for:
- dataset analysis,
- model training and evaluation,
- single-image and batch directory inference,
- webcam live recognition.

## Key features

- **Multiple recognizers**: LBPH, Eigenfaces, Fisherfaces.
- **Unknown-person rejection**: configurable confidence threshold to label low-confidence predictions as `Unknown`.
- **Dataset analyzer**: reports class balance and warns about classes with too few samples.
- **Evaluation artifacts**: confusion matrix image + JSON report.
- **Model card export**: creates `MODEL_CARD.md` with training metadata.
- **Batch inference**: run prediction over an image directory and export JSON.
- **Optional annotated outputs**: save predicted images with boxes/labels.

## Installation

```bash
pip install -r requirements.txt
```

> Important: `opencv-contrib-python` is required because this project uses `cv2.face.*` APIs.

## Project files

- `face_recognition_system.py` – core recognizer + CLI
- `system_test_script.py` – smoke checks
- `requirements.txt` – dependencies
- `package_project.py` – create distributable ZIP archive
- `web_interface.py` – Streamlit web UI

## Dataset formats

### 1) Person folders (recommended)

```
dataset/
  alice/
    img1.jpg
    img2.jpg
  bob/
    img1.jpg
```

Use `--structure person_folders`.

### 2) Flat filenames with labels

```
dataset/
  alice_1.jpg
  alice_2.jpg
  bob_1.jpg
```

Use `--structure flat_with_labels`.

### 3) CSV labeled

```
dataset/
  labels.csv
  image1.jpg
  image2.jpg
```

`labels.csv` must have columns: `image,person`

## CLI usage

### 1) Analyze dataset quality

```bash
python face_recognition_system.py \
  --analyze-dataset sample_test_faces \
  --structure person_folders
```

### 2) Train model

```bash
python face_recognition_system.py \
  --train \
  --dataset sample_test_faces \
  --structure person_folders \
  --model lbph \
  --validation-split 0.2 \
  --unknown-threshold 55
```

### 3) Evaluate model

```bash
python face_recognition_system.py \
  --evaluate sample_test_faces \
  --structure person_folders \
  --output-json reports/test_eval.json
```

### 4) Predict one image

```bash
python face_recognition_system.py \
  --predict-image my_test.jpg \
  --save-annotated \
  --annotated-dir outputs/annotated \
  --output-json outputs/single_prediction.json
```

### 5) Predict a full directory

```bash
python face_recognition_system.py \
  --predict-dir incoming_images \
  --save-annotated \
  --annotated-dir outputs/annotated \
  --output-json outputs/batch_predictions.json
```

### 6) Live recognition

```bash
python face_recognition_system.py --live --camera-id 0
```


### 7) Package project as ZIP

```bash
python package_project.py
```

This generates `Face-Recognition-System.zip` with core project files.


### 8) Run web interface (browser UI)

```bash
streamlit run web_interface.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## Smoke test

```bash
python system_test_script.py
```

Optional webcam checks:

```bash
python system_test_script.py --with-camera
```

## “Is it working?” quick check

```bash
python face_recognition_system.py --help
python system_test_script.py
```

If dependencies are installed, help should render and smoke tests should pass. Camera checks are optional and depend on hardware availability.

## Real-world next upgrades (optional)

- add liveness detection / anti-spoofing,
- switch to embedding-based recognition (ArcFace/FaceNet) for large-scale identity search,
- add API service (FastAPI) for production integration,
- add data governance, consent, and auditing pipeline.
