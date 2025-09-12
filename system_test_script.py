#!/usr/bin/env python3
"""
Face Recognition System Test Script
This script performs comprehensive tests to verify your system is working correctly.
"""

import sys
import time
import os
import cv2
import numpy as np


def test_dependencies():
    """Test if all required dependencies are installed"""
    print("=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)

    dependencies = [
        ('opencv-python', 'cv2'),
        ('face-recognition', 'face_recognition'),
        ('numpy', 'numpy'),
        ('pickle', 'pickle')
    ]

    all_good = True

    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {package_name}: OK")
        except ImportError:
            print(f"❌ {package_name}: NOT FOUND")
            print(f"   Install with: pip install {package_name}")
            all_good = False

    return all_good


def test_camera():
    """Test camera functionality"""
    print("\n" + "=" * 60)
    print("TESTING CAMERA")
    print("=" * 60)

    camera_found = False
    working_cameras = []

    # Test multiple camera indices
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()

            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Camera {i}: Working ({width}x{height})")
                working_cameras.append(i)
                camera_found = True
            else:
                print(f"❌ Camera {i}: No signal")

            cap.release()

        except Exception as e:
            print(f"❌ Camera {i}: Error - {str(e)}")

    if camera_found:
        print(f"\n📷 Found {len(working_cameras)} working camera(s): {working_cameras}")
        return working_cameras[0]  # Return first working camera
    else:
        print("\n⚠️  No working cameras found!")
        return None


def test_face_detection(camera_id=0):
    """Test basic face detection"""
    print("\n" + "=" * 60)
    print("TESTING FACE DETECTION")
    print("=" * 60)

    try:
        import face_recognition

        # Try to open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return False

        print("📸 Starting face detection test...")
        print("   Position your face in front of the camera")
        print("   Press 's' to test detection, 'q' to quit")

        face_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)

            # Draw rectangles around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected!", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                face_detected = True

            # Show frame
            cv2.imshow('Face Detection Test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and face_detected:
                print("✅ Face detection working!")
                break

        cap.release()
        cv2.destroyAllWindows()

        if face_detected:
            print("✅ Face detection: WORKING")
            return True
        else:
            print("⚠️  No faces were detected during test")
            return False

    except Exception as e:
        print(f"❌ Face detection test failed: {str(e)}")
        return False


def performance_test(camera_id=0):
    """Test system performance"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE")
    print("=" * 60)

    try:
        import face_recognition

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return False

        print("⏱️  Running 30-second performance test...")

        frame_count = 0
        face_detections = 0
        start_time = time.time()
        test_duration = 10  # 10 seconds test

        while True:
            current_time = time.time()
            if current_time - start_time > test_duration:
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Resize frame for faster processing (like the main system)
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces every 3rd frame (like the main system)
            if frame_count % 3 == 0:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if face_locations:
                    face_detections += 1

        cap.release()

        # Calculate metrics
        total_time = time.time() - start_time
        fps = frame_count / total_time

        print(f"📊 Performance Results:")
        print(f"   Total frames: {frame_count}")
        print(f"   Test duration: {total_time:.1f} seconds")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Face detections: {face_detections}")

        # Performance evaluation
        if fps >= 15:
            print("✅ Performance: EXCELLENT")
        elif fps >= 10:
            print("✅ Performance: GOOD")
        elif fps >= 5:
            print("⚠️  Performance: ACCEPTABLE (but slow)")
        else:
            print("❌ Performance: POOR (needs optimization)")

        return fps >= 5  # Minimum acceptable performance

    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False


def test_main_system():
    """Test the main face recognition system"""
    print("\n" + "=" * 60)
    print("TESTING MAIN SYSTEM")
    print("=" * 60)

    try:
        # Try to import the main system
        from face_recognition_system import FaceRecognitionSystem

        print("✅ Main system imports successfully")

        # Try to initialize the system
        face_system = FaceRecognitionSystem()
        print("✅ System initializes successfully")

        # Check if database methods work
        initial_count = len(face_system.known_face_names)
        print(f"✅ Face database loaded ({initial_count} faces)")

        return True

    except ImportError as e:
        print(f"❌ Cannot import main system: {str(e)}")
        print("   Make sure face_recognition_system.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        return False


def create_sample_test_faces():
    """Create sample test directory structure"""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE TEST SETUP")
    print("=" * 60)

    test_dir = "sample_test_faces"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"✅ Created directory: {test_dir}")

        readme_content = """
# Sample Test Faces Directory

To test face recognition, add face images to this directory:

1. Add clear face photos (JPG, PNG format)
2. Name files with person's name (e.g., 'john_doe.jpg')
3. One face per image works best
4. Good lighting and front-facing photos recommended

Example:
- john_doe.jpg
- jane_smith.png  
- bob_wilson.jpeg

Then run:
python face_recognition_system.py --register-dir sample_test_faces/
"""

        with open(os.path.join(test_dir, "README.txt"), 'w') as f:
            f.write(readme_content)

        print(f"✅ Created README in {test_dir}")
        print(f"📝 Add face images to {test_dir}/ for testing")
    else:
        print(f"✅ Directory {test_dir} already exists")


def main():
    """Run all tests"""
    print("🚀 FACE RECOGNITION SYSTEM TEST SUITE")
    print("This will test all components of your face recognition system\n")

    all_tests_passed = True

    # Test 1: Dependencies
    if not test_dependencies():
        all_tests_passed = False
        print("\n❌ Dependency test failed. Please install missing packages.")
        return

    # Test 2: Camera
    camera_id = test_camera()
    if camera_id is None:
        all_tests_passed = False
        print("\n❌ No working camera found. Cannot proceed with video tests.")
    else:
        # Test 3: Face Detection (only if camera works)
        if not test_face_detection(camera_id):
            all_tests_passed = False

        # Test 4: Performance (only if camera works)
        if not performance_test(camera_id):
            all_tests_passed = False

    # Test 5: Main System
    if not test_main_system():
        all_tests_passed = False

    # Test 6: Create sample setup
    create_sample_test_faces()

    # Final Results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour face recognition system is ready to use!")
        print("\nNext steps:")
        print("1. Add face images to 'sample_test_faces/' directory")
        print("2. Run: python face_recognition_system.py --register-dir sample_test_faces/")
        print("3. Test real-time recognition: python face_recognition_system.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before using the system.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
