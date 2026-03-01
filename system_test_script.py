
-import sys
-import time
-import os
-import cv2
-import numpy as np
-
-
-def test_dependencies():
-    """Test if all required dependencies are installed"""
-    print("=" * 60)
-    print("TESTING DEPENDENCIES")
-    print("=" * 60)
-
-    dependencies = [
-        ('opencv-python', 'cv2'),
-        ('face-recognition', 'face_recognition'),
-        ('numpy', 'numpy'),
-        ('pickle', 'pickle')
-    ]
-
-    all_good = True
-
-    for package_name, import_name in dependencies:
-        try:
-            __import__(import_name)
-            print(f"✅ {package_name}: OK")
-        except ImportError:
-            print(f"❌ {package_name}: NOT FOUND")
-            print(f"   Install with: pip install {package_name}")
-            all_good = False
-
-    return all_good
-
-
-def test_camera():
-    """Test camera functionality"""
-    print("\n" + "=" * 60)
-    print("TESTING CAMERA")
-    print("=" * 60)
-
-    camera_found = False
-    working_cameras = []
-
-    # Test multiple camera indices
-    for i in range(5):
-        try:
-            cap = cv2.VideoCapture(i)
-            ret, frame = cap.read()
-
-            if ret and frame is not None:
-                height, width = frame.shape[:2]
-                print(f"✅ Camera {i}: Working ({width}x{height})")
-                working_cameras.append(i)
-                camera_found = True
-            else:
-                print(f"❌ Camera {i}: No signal")
-
-            cap.release()
-
-        except Exception as e:
-            print(f"❌ Camera {i}: Error - {str(e)}")
-
-    if camera_found:
-        print(f"\n📷 Found {len(working_cameras)} working camera(s): {working_cameras}")
-        return working_cameras[0]  # Return first working camera
-    else:
-        print("\n⚠️  No working cameras found!")
-        return None
-
-
-def test_face_detection(camera_id=0):
-    """Test basic face detection"""
-    print("\n" + "=" * 60)
-    print("TESTING FACE DETECTION")
-    print("=" * 60)
-
-    try:
-        import face_recognition
-
-        # Try to open camera
-        cap = cv2.VideoCapture(camera_id)
-        if not cap.isOpened():
-            print(f"❌ Cannot open camera {camera_id}")
-            return False
-
-        print("📸 Starting face detection test...")
-        print("   Position your face in front of the camera")
-        print("   Press 's' to test detection, 'q' to quit")
-
-        face_detected = False
-
-        while True:
-            ret, frame = cap.read()
-            if not ret:
-                break
-
-            # Convert BGR to RGB
-            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
-
-            # Detect faces
-            face_locations = face_recognition.face_locations(rgb_frame)
-
-            # Draw rectangles around faces
-            for (top, right, bottom, left) in face_locations:
-                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
-                cv2.putText(frame, "Face Detected!", (left, top - 10),
-                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
-                face_detected = True
-
-            # Show frame
-            cv2.imshow('Face Detection Test', frame)
-
-            key = cv2.waitKey(1) & 0xFF
-            if key == ord('q'):
-                break
-            elif key == ord('s') and face_detected:
-                print("✅ Face detection working!")
-                break
-
-        cap.release()
-        cv2.destroyAllWindows()
-
-        if face_detected:
-            print("✅ Face detection: WORKING")
-            return True
-        else:
-            print("⚠️  No faces were detected during test")
-            return False
-
-    except Exception as e:
-        print(f"❌ Face detection test failed: {str(e)}")
-        return False
-
-
-def performance_test(camera_id=0):
-    """Test system performance"""
-    print("\n" + "=" * 60)
-    print("TESTING PERFORMANCE")
-    print("=" * 60)
-
-    try:
-        import face_recognition
-
-        cap = cv2.VideoCapture(camera_id)
-        if not cap.isOpened():
-            print(f"❌ Cannot open camera {camera_id}")
-            return False
-
-        print("⏱️  Running 30-second performance test...")
-
-        frame_count = 0
-        face_detections = 0
-        start_time = time.time()
-        test_duration = 10  # 10 seconds test
-
-        while True:
-            current_time = time.time()
-            if current_time - start_time > test_duration:
-                break
-
-            ret, frame = cap.read()
-            if not ret:
-                continue
-
-            frame_count += 1
-
-            # Resize frame for faster processing (like the main system)
-            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
-            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
-
-            # Detect faces every 3rd frame (like the main system)
-            if frame_count % 3 == 0:
-                face_locations = face_recognition.face_locations(rgb_small_frame)
-                if face_locations:
-                    face_detections += 1
-
-        cap.release()
-
-        # Calculate metrics
-        total_time = time.time() - start_time
-        fps = frame_count / total_time
-
-        print(f"📊 Performance Results:")
-        print(f"   Total frames: {frame_count}")
-        print(f"   Test duration: {total_time:.1f} seconds")
-        print(f"   Average FPS: {fps:.1f}")
-        print(f"   Face detections: {face_detections}")
-
-        # Performance evaluation
-        if fps >= 15:
-            print("✅ Performance: EXCELLENT")
-        elif fps >= 10:
-            print("✅ Performance: GOOD")
-        elif fps >= 5:
-            print("⚠️  Performance: ACCEPTABLE (but slow)")
-        else:
-            print("❌ Performance: POOR (needs optimization)")
-
-        return fps >= 5  # Minimum acceptable performance
-
-    except Exception as e:
-        print(f"❌ Performance test failed: {str(e)}")
-        return False
-
-
-def test_main_system():
-    """Test the main face recognition system"""
-    print("\n" + "=" * 60)
-    print("TESTING MAIN SYSTEM")
-    print("=" * 60)
-
-    try:
-        # Try to import the main system
-        from face_recognition_system import FaceRecognitionSystem
-
-        print("✅ Main system imports successfully")
-
-        # Try to initialize the system
-        face_system = FaceRecognitionSystem()
-        print("✅ System initializes successfully")
-
-        # Check if database methods work
-        initial_count = len(face_system.known_face_names)
-        print(f"✅ Face database loaded ({initial_count} faces)")
-
-        return True
-
-    except ImportError as e:
-        print(f"❌ Cannot import main system: {str(e)}")
-        print("   Make sure face_recognition_system.py is in the same directory")
-        return False
-    except Exception as e:
-        print(f"❌ System initialization failed: {str(e)}")
-        return False
-
-
-def create_sample_test_faces():
-    """Create sample test directory structure"""
-    print("\n" + "=" * 60)
-    print("CREATING SAMPLE TEST SETUP")
-    print("=" * 60)
-
-    test_dir = "sample_test_faces"
-
-    if not os.path.exists(test_dir):
-        os.makedirs(test_dir)
-        print(f"✅ Created directory: {test_dir}")
-
-        readme_content = """
-# Sample Test Faces Directory
-
-To test face recognition, add face images to this directory:
-
-1. Add clear face photos (JPG, PNG format)
-2. Name files with person's name (e.g., 'john_doe.jpg')
-3. One face per image works best
-4. Good lighting and front-facing photos recommended
-
-Example:
-- john_doe.jpg
-- jane_smith.png  
-- bob_wilson.jpeg
-
-Then run:
-python face_recognition_system.py --register-dir sample_test_faces/
-"""
-
-        with open(os.path.join(test_dir, "README.txt"), 'w') as f:
-            f.write(readme_content)
-
-        print(f"✅ Created README in {test_dir}")
-        print(f"📝 Add face images to {test_dir}/ for testing")
-    else:
-        print(f"✅ Directory {test_dir} already exists")
-
-
-def main():
-    """Run all tests"""
-    print("🚀 FACE RECOGNITION SYSTEM TEST SUITE")
-    print("This will test all components of your face recognition system\n")
-
-    all_tests_passed = True
-
-    # Test 1: Dependencies
-    if not test_dependencies():
-        all_tests_passed = False
-        print("\n❌ Dependency test failed. Please install missing packages.")
-        return
-
-    # Test 2: Camera
-    camera_id = test_camera()
-    if camera_id is None:
-        all_tests_passed = False
-        print("\n❌ No working camera found. Cannot proceed with video tests.")
-    else:
-        # Test 3: Face Detection (only if camera works)
-        if not test_face_detection(camera_id):
-            all_tests_passed = False
-
-        # Test 4: Performance (only if camera works)
-        if not performance_test(camera_id):
-            all_tests_passed = False
-
-    # Test 5: Main System
-    if not test_main_system():
-        all_tests_passed = False
-
-    # Test 6: Create sample setup
-    create_sample_test_faces()
-
-    # Final Results
-    print("\n" + "=" * 60)
-    print("FINAL TEST RESULTS")
-    print("=" * 60)
-
-    if all_tests_passed:
-        print("🎉 ALL TESTS PASSED!")
-        print("\nYour face recognition system is ready to use!")
-        print("\nNext steps:")
-        print("1. Add face images to 'sample_test_faces/' directory")
-        print("2. Run: python face_recognition_system.py --register-dir sample_test_faces/")
-        print("3. Test real-time recognition: python face_recognition_system.py")
-    else:
-        print("❌ SOME TESTS FAILED")
-        print("\nPlease fix the issues above before using the system.")
-
-    print("\n" + "=" * 60)
-
-
-if __name__ == "__main__":
-    main()
+#!/usr/bin/env python3
+"""
+Face Recognition System smoke tests.
+
+Runs non-interactive checks by default so it can be used in CI/dev containers.
+Use --with-camera to include camera and runtime performance checks.
+"""
+
+import argparse
+import importlib
+import subprocess
+import sys
+import time
+from pathlib import Path
+
+
+def _load_cv2():
+    try:
+        return importlib.import_module("cv2")
+    except ImportError:
+        return None
+
+
+def test_dependencies():
+    """Test if required dependencies are installed/importable."""
+    print("=" * 60)
+    print("TESTING DEPENDENCIES")
+    print("=" * 60)
+
+    dependencies = [
+        ("opencv-contrib-python", "cv2", True),
+        ("numpy", "numpy", True),
+        ("scikit-learn", "sklearn", True),
+        ("matplotlib", "matplotlib", True),
+        ("seaborn", "seaborn", True),
+        ("pandas", "pandas", False),
+        ("streamlit", "streamlit", False),
+    ]
+
+    all_required_ok = True
+
+    for package_name, import_name, required in dependencies:
+        try:
+            importlib.import_module(import_name)
+            print(f"✅ {package_name}: OK")
+        except ImportError:
+            label = "❌" if required else "⚠️"
+            print(f"{label} {package_name}: NOT FOUND")
+            print(f"   Install with: pip install {package_name}")
+            if required:
+                all_required_ok = False
+
+    return all_required_ok
+
+
+def test_project_files():
+    """Validate expected project files exist."""
+    print("\n" + "=" * 60)
+    print("TESTING PROJECT FILES")
+    print("=" * 60)
+
+    required_paths = [
+        Path("face_recognition_system.py"),
+        Path("system_test_script.py"),
+        Path("README.md"),
+        Path("requirements.txt"),
+        Path("web_interface.py"),
+    ]
+
+    all_present = True
+    for path in required_paths:
+        if path.exists():
+            print(f"✅ Found {path}")
+        else:
+            print(f"❌ Missing {path}")
+            all_present = False
+
+    return all_present
+
+
+def test_cli_help():
+    """Check that CLI entrypoint responds and shows major flags."""
+    print("\n" + "=" * 60)
+    print("TESTING CLI")
+    print("=" * 60)
+
+    proc = subprocess.run(
+        ["python", "face_recognition_system.py", "--help"],
+        capture_output=True,
+        text=True,
+        check=False,
+    )
+
+    expected_flags = ["--predict-dir", "--analyze-dataset", "--save-annotated", "--unknown-threshold"]
+    if proc.returncode == 0 and all(flag in proc.stdout for flag in expected_flags):
+        print("✅ CLI help works and includes key feature flags")
+        return True
+
+    print("❌ CLI help failed")
+    if proc.stderr:
+        print(proc.stderr)
+    return False
+
+
+def test_main_system():
+    """Import and instantiate the real main class."""
+    print("\n" + "=" * 60)
+    print("TESTING MAIN SYSTEM")
+    print("=" * 60)
+
+    try:
+        from face_recognition_system import TrainableFaceRecognizer
+
+        recognizer = TrainableFaceRecognizer(data_dir="test_training_data")
+        print("✅ TrainableFaceRecognizer imports and initializes")
+        print(f"✅ Model type: {recognizer.model_type}")
+        return True
+    except Exception as exc:
+        print(f"❌ Main system failed: {exc}")
+        return False
+
+
+def test_camera(max_indexes=3):
+    """Try to find a working camera index."""
+    cv2 = _load_cv2()
+    if cv2 is None:
+        print("⚠️ Skipping camera test: cv2 is not installed")
+        return None
+
+    print("\n" + "=" * 60)
+    print("TESTING CAMERA")
+    print("=" * 60)
+
+    working_cameras = []
+    for index in range(max_indexes):
+        cap = cv2.VideoCapture(index)
+        try:
+            ok, frame = cap.read()
+            if ok and frame is not None:
+                h, w = frame.shape[:2]
+                print(f"✅ Camera {index}: Working ({w}x{h})")
+                working_cameras.append(index)
+            else:
+                print(f"⚠️ Camera {index}: Not available")
+        finally:
+            cap.release()
+
+    if not working_cameras:
+        print("⚠️ No working camera found")
+        return None
+
+    print(f"📷 Found cameras: {working_cameras}")
+    return working_cameras[0]
+
+
+def performance_test(camera_id=0, duration_s=5):
+    """Basic FPS test with haar-cascade face detection."""
+    cv2 = _load_cv2()
+    if cv2 is None:
+        print("⚠️ Skipping performance test: cv2 is not installed")
+        return False
+
+    print("\n" + "=" * 60)
+    print("TESTING PERFORMANCE")
+    print("=" * 60)
+
+    cap = cv2.VideoCapture(camera_id)
+    if not cap.isOpened():
+        print(f"⚠️ Cannot open camera {camera_id}")
+        return False
+
+    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
+
+    frame_count = 0
+    detection_count = 0
+    start = time.time()
+
+    while time.time() - start < duration_s:
+        ok, frame = cap.read()
+        if not ok:
+            continue
+
+        frame_count += 1
+        if frame_count % 3 == 0:
+            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
+            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
+            if len(faces) > 0:
+                detection_count += 1
+
+    cap.release()
+
+    elapsed = time.time() - start
+    fps = frame_count / elapsed if elapsed > 0 else 0
+
+    print("📊 Performance Results:")
+    print(f"   Frames: {frame_count}")
+    print(f"   Duration: {elapsed:.1f}s")
+    print(f"   FPS: {fps:.1f}")
+    print(f"   Detection frames: {detection_count}")
+
+    if fps >= 10:
+        print("✅ Performance: GOOD")
+        return True
+    if fps >= 5:
+        print("⚠️ Performance: ACCEPTABLE")
+        return True
+
+    print("❌ Performance: POOR")
+    return False
+
+
+def create_sample_test_faces():
+    """Create sample dataset structure with documentation."""
+    print("\n" + "=" * 60)
+    print("CREATING SAMPLE TEST SETUP")
+    print("=" * 60)
+
+    root = Path("sample_test_faces")
+    (root / "person_a").mkdir(parents=True, exist_ok=True)
+    (root / "person_b").mkdir(parents=True, exist_ok=True)
+
+    readme = root / "README.txt"
+    readme.write_text(
+        """# Sample Training Dataset
+
+Add JPG/PNG images into person-specific subfolders, e.g.:
+sample_test_faces/person_a/img1.jpg
+sample_test_faces/person_b/img1.jpg
+
+Then train with:
+python face_recognition_system.py --train --dataset sample_test_faces --structure person_folders
+""",
+        encoding="utf-8",
+    )
+
+    print(f"✅ Dataset scaffold ready at {root}")
+    return True
+
+
+def main():
+    parser = argparse.ArgumentParser(description="Face Recognition System Test Suite")
+    parser.add_argument("--with-camera", action="store_true", help="Include camera tests")
+    args = parser.parse_args()
+
+    all_passed = True
+
+    if not test_project_files():
+        all_passed = False
+
+    if not test_dependencies():
+        all_passed = False
+
+    if not test_main_system():
+        all_passed = False
+
+    if not test_cli_help():
+        all_passed = False
+
+    if not create_sample_test_faces():
+        all_passed = False
+
+    if args.with_camera:
+        cam = test_camera()
+        if cam is not None and not performance_test(cam):
+            all_passed = False
+
+    print("\n" + "=" * 60)
+    print("FINAL TEST RESULTS")
+    print("=" * 60)
+    print("🎉 ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
+    sys.exit(0 if all_passed else 1)
+
+
+if __name__ == "__main__":
+    main()
