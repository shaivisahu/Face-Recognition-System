#!/usr/bin/env python3
"""
Face Recognition System smoke tests.

Runs non-interactive checks by default so it can be used in CI/dev containers.
Use --with-camera to include camera and runtime performance checks.
"""

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path


def _load_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError:
        return None


def test_dependencies():
    """Test if required dependencies are installed/importable."""
    print("=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)

    dependencies = [
        ("opencv-contrib-python", "cv2", True),
        ("numpy", "numpy", True),
        ("scikit-learn", "sklearn", True),
        ("matplotlib", "matplotlib", True),
        ("seaborn", "seaborn", True),
        ("pandas", "pandas", False),
        ("streamlit", "streamlit", False),
    ]

    all_required_ok = True

    for package_name, import_name, required in dependencies:
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}: OK")
        except ImportError:
            label = "❌" if required else "⚠️"
            print(f"{label} {package_name}: NOT FOUND")
            print(f"   Install with: pip install {package_name}")
            if required:
                all_required_ok = False

    return all_required_ok


def test_project_files():
    """Validate expected project files exist."""
    print("\n" + "=" * 60)
    print("TESTING PROJECT FILES")
    print("=" * 60)

    required_paths = [
        Path("face_recognition_system.py"),
        Path("system_test_script.py"),
        Path("README.md"),
        Path("requirements.txt"),
        Path("web_interface.py"),
    ]

    all_present = True
    for path in required_paths:
        if path.exists():
            print(f"✅ Found {path}")
        else:
            print(f"❌ Missing {path}")
            all_present = False

    return all_present


def test_cli_help():
    """Check that CLI entrypoint responds and shows major flags."""
    print("\n" + "=" * 60)
    print("TESTING CLI")
    print("=" * 60)

    proc = subprocess.run(
        ["python", "face_recognition_system.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    expected_flags = ["--predict-dir", "--analyze-dataset", "--save-annotated", "--unknown-threshold"]
    if proc.returncode == 0 and all(flag in proc.stdout for flag in expected_flags):
        print("✅ CLI help works and includes key feature flags")
        return True

    print("❌ CLI help failed")
    if proc.stderr:
        print(proc.stderr)
    return False


def test_main_system():
    """Import and instantiate the real main class."""
    print("\n" + "=" * 60)
    print("TESTING MAIN SYSTEM")
    print("=" * 60)

    try:
        from face_recognition_system import TrainableFaceRecognizer

        recognizer = TrainableFaceRecognizer(data_dir="test_training_data")
        print("✅ TrainableFaceRecognizer imports and initializes")
        print(f"✅ Model type: {recognizer.model_type}")
        return True
    except Exception as exc:
        print(f"❌ Main system failed: {exc}")
        return False


def test_camera(max_indexes=3):
    """Try to find a working camera index."""
    cv2 = _load_cv2()
    if cv2 is None:
        print("⚠️ Skipping camera test: cv2 is not installed")
        return None

    print("\n" + "=" * 60)
    print("TESTING CAMERA")
    print("=" * 60)

    working_cameras = []
    for index in range(max_indexes):
        cap = cv2.VideoCapture(index)
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ Camera {index}: Working ({w}x{h})")
                working_cameras.append(index)
            else:
                print(f"⚠️ Camera {index}: Not available")
        finally:
            cap.release()

    if not working_cameras:
        print("⚠️ No working camera found")
        return None

    print(f"📷 Found cameras: {working_cameras}")
    return working_cameras[0]


def performance_test(camera_id=0, duration_s=5):
    """Basic FPS test with haar-cascade face detection."""
    cv2 = _load_cv2()
    if cv2 is None:
        print("⚠️ Skipping performance test: cv2 is not installed")
        return False

    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"⚠️ Cannot open camera {camera_id}")
        return False

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_count = 0
    detection_count = 0
    start = time.time()

    while time.time() - start < duration_s:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_count += 1
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            if len(faces) > 0:
                detection_count += 1

    cap.release()

    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0

    print("📊 Performance Results:")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {elapsed:.1f}s")
    print(f"   FPS: {fps:.1f}")
    print(f"   Detection frames: {detection_count}")

    if fps >= 10:
        print("✅ Performance: GOOD")
        return True
    if fps >= 5:
        print("⚠️ Performance: ACCEPTABLE")
        return True

    print("❌ Performance: POOR")
    return False


def create_sample_test_faces():
    """Create sample dataset structure with documentation."""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE TEST SETUP")
    print("=" * 60)

    root = Path("sample_test_faces")
    (root / "person_a").mkdir(parents=True, exist_ok=True)
    (root / "person_b").mkdir(parents=True, exist_ok=True)

    readme = root / "README.txt"
    readme.write_text(
        """# Sample Training Dataset

Add JPG/PNG images into person-specific subfolders, e.g.:
sample_test_faces/person_a/img1.jpg
sample_test_faces/person_b/img1.jpg

Then train with:
python face_recognition_system.py --train --dataset sample_test_faces --structure person_folders
""",
        encoding="utf-8",
    )

    print(f"✅ Dataset scaffold ready at {root}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Face Recognition System Test Suite")
    parser.add_argument("--with-camera", action="store_true", help="Include camera tests")
    args = parser.parse_args()

    all_passed = True

    if not test_project_files():
        all_passed = False

    if not test_dependencies():
        all_passed = False

    if not test_main_system():
        all_passed = False

    if not test_cli_help():
        all_passed = False

    if not create_sample_test_faces():
        all_passed = False

    if args.with_camera:
        cam = test_camera()
        if cam is not None and not performance_test(cam):
            all_passed = False

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print("🎉 ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
