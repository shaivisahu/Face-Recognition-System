
import cv2
import numpy as np
import os
import pickle
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse
from pathlib import Path


class TrainableFaceRecognizer:
    def __init__(self, model_type='lbph', data_dir='training_data'):
        """
        Initialize trainable face recognizer

        Args:
            model_type: 'lbph', 'eigenfaces', or 'fisherfaces'
            data_dir: Directory to store training data and models
        """
        self.model_type = model_type.lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Initialize recognizer based on type
        if self.model_type == 'lbph':
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        elif self.model_type == 'eigenfaces':
            self.recognizer = cv2.face.EigenFaceRecognizer_create()
        elif self.model_type == 'fisherfaces':
            self.recognizer = cv2.face.FisherFaceRecognizer_create()
        else:
            raise ValueError("model_type must be 'lbph', 'eigenfaces', or 'fisherfaces'")

        # Training data storage
        self.faces = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.training_metadata = {}

        # Model files
        self.model_file = self.data_dir / f"{self.model_type}_model.yml"
        self.labels_file = self.data_dir / "labels.pkl"
        self.metadata_file = self.data_dir / "training_metadata.json"

        self.is_trained = False

        print(f"Initialized {self.model_type.upper()} face recognizer")
        self.load_model()

    def preprocess_face(self, face_img, target_size=(200, 200)):
        """Preprocess face image"""
        if face_img is None:
            return None

        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        face_img = cv2.resize(face_img, target_size)

        # Histogram equalization for better lighting normalization
        face_img = cv2.equalizeHist(face_img)

        return face_img

    def extract_faces_from_image(self, image_path, person_name=None):
        """Extract faces from a single image"""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load image: {image_path}")
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        extracted_faces = []
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = self.preprocess_face(face_img)

            if face_img is not None:
                extracted_faces.append({
                    'image': face_img,
                    'person': person_name or 'unknown',
                    'source': str(image_path),
                    'bbox': (x, y, w, h)
                })

        return extracted_faces

    def load_dataset_from_directory(self, dataset_path, structure_type='person_folders'):
        """
        Load dataset from directory

        Supported structures:
        1. 'person_folders': dataset/person1/img1.jpg, dataset/person2/img2.jpg
        2. 'flat_with_labels': dataset/person1_img1.jpg, dataset/person2_img2.jpg
        3. 'csv_labeled': images in folder + CSV with image,person mappings
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path {dataset_path} does not exist")

        print(f"Loading dataset from {dataset_path} with structure: {structure_type}")

        if structure_type == 'person_folders':
            return self._load_person_folders(dataset_path)
        elif structure_type == 'flat_with_labels':
            return self._load_flat_labeled(dataset_path)
        elif structure_type == 'csv_labeled':
            return self._load_csv_labeled(dataset_path)
        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")

    def _load_person_folders(self, dataset_path):
        """Load from person folders structure"""
        training_data = []

        for person_dir in dataset_path.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            print(f"Processing {person_name}...")

            image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')) + \
                          list(person_dir.glob('*.jpeg')) + list(person_dir.glob('*.JPG'))

            for img_file in image_files:
                faces = self.extract_faces_from_image(img_file, person_name)
                training_data.extend(faces)

                if len(faces) == 0:
                    print(f"  No faces found in {img_file}")
                else:
                    print(f"  Extracted {len(faces)} face(s) from {img_file.name}")

        return training_data

    def _load_flat_labeled(self, dataset_path):
        """Load from flat directory with person names in filenames"""
        training_data = []

        image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png')) + \
                      list(dataset_path.glob('*.jpeg'))

        for img_file in image_files:
            # Extract person name from filename (before first underscore or number)
            person_name = img_file.stem.split('_')[0].split('0')[0].split('1')[0]
            person_name = person_name.split('2')[0].split('3')[0].split('4')[0]
            person_name = person_name.split('5')[0].split('6')[0].split('7')[0]
            person_name = person_name.split('8')[0].split('9')[0]

            faces = self.extract_faces_from_image(img_file, person_name)
            training_data.extend(faces)

            if len(faces) > 0:
                print(f"Extracted {len(faces)} face(s) from {img_file.name} -> {person_name}")

        return training_data

    def _load_csv_labeled(self, dataset_path):
        """Load from CSV with image,person mappings"""
        import pandas as pd

        csv_file = dataset_path / "labels.csv"
        if not csv_file.exists():
            raise ValueError(f"CSV file {csv_file} not found")

        df = pd.read_csv(csv_file)
        training_data = []

        for _, row in df.iterrows():
            img_path = dataset_path / row['image']
            person_name = row['person']

            if img_path.exists():
                faces = self.extract_faces_from_image(img_path, person_name)
                training_data.extend(faces)

                if len(faces) > 0:
                    print(f"Extracted {len(faces)} face(s) from {row['image']} -> {person_name}")

        return training_data

    def prepare_training_data(self, training_data, validation_split=0.2):
        """Prepare training data for model training"""
        if not training_data:
            raise ValueError("No training data provided")

        print(f"Preparing {len(training_data)} face samples...")

        # Organize data by person
        person_data = defaultdict(list)
        for face_data in training_data:
            person_data[face_data['person']].append(face_data)

        # Assign labels to persons
        current_label = 0
        for person_name in sorted(person_data.keys()):
            if person_name not in self.name_to_label:
                self.name_to_label[person_name] = current_label
                self.label_to_name[current_label] = person_name
                current_label += 1

        # Prepare faces and labels arrays
        faces = []
        labels = []
        metadata = []

        for person_name, face_list in person_data.items():
            person_label = self.name_to_label[person_name]

            for face_data in face_list:
                faces.append(face_data['image'])
                labels.append(person_label)
                metadata.append({
                    'person': person_name,
                    'source': face_data['source'],
                    'bbox': face_data['bbox']
                })

        # Convert to numpy arrays
        faces = np.array(faces)
        labels = np.array(labels)

        # Split into train and validation sets
        if validation_split > 0:
            (train_faces, val_faces, train_labels, val_labels,
             train_metadata, val_metadata) = train_test_split(
                faces, labels, metadata, test_size=validation_split,
                random_state=42, stratify=labels
            )
        else:
            train_faces, train_labels, train_metadata = faces, labels, metadata
            val_faces, val_labels, val_metadata = None, None, None

        print(f"Training set: {len(train_faces)} samples")
        if val_faces is not None:
            print(f"Validation set: {len(val_faces)} samples")

        print("Class distribution:")
        for label, count in Counter(train_labels).items():
            print(f"  {self.label_to_name[label]}: {count} samples")

        return {
            'train': (train_faces, train_labels, train_metadata),
            'validation': (val_faces, val_labels, val_metadata) if val_faces is not None else None
        }

    def train_model(self, dataset_path, structure_type='person_folders', validation_split=0.2):
        """Train the face recognition model"""
        print("=" * 60)
        print("TRAINING FACE RECOGNITION MODEL")
        print("=" * 60)

        # Load dataset
        training_data = self.load_dataset_from_directory(dataset_path, structure_type)

        if not training_data:
            raise ValueError("No training data found")

        print(f"Found {len(training_data)} face samples")

        # Prepare data
        data = self.prepare_training_data(training_data, validation_split)
        train_faces, train_labels, train_metadata = data['train']

        # Train the model
        print("Training model...")
        start_time = time.time()

        self.recognizer.train(train_faces, train_labels)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Store training metadata
        self.training_metadata = {
            'model_type': self.model_type,
            'num_samples': len(train_faces),
            'num_classes': len(self.label_to_name),
            'training_time': training_time,
            'validation_split': validation_split,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        self.is_trained = True

        # Evaluate on validation set if available
        if data['validation'] is not None:
            val_faces, val_labels, val_metadata = data['validation']
            self.evaluate_model(val_faces, val_labels, "Validation Set")

        # Save model
        self.save_model()

        print("Training completed successfully!")
        return self.training_metadata

    def evaluate_model(self, test_faces, test_labels, dataset_name="Test Set"):
        """Evaluate model performance"""
        if not self.is_trained:
            print("Model not trained yet")
            return None

        print(f"\nEvaluating on {dataset_name}...")

        predictions = []
        confidences = []

        for face in test_faces:
            label, confidence = self.recognizer.predict(face)
            predictions.append(label)
            confidences.append(confidence)

        # Calculate accuracy
        correct = np.sum(predictions == test_labels)
        accuracy = correct / len(test_labels)

        print(f"Accuracy: {accuracy:.3f} ({correct}/{len(test_labels)})")
        print(f"Average confidence: {np.mean(confidences):.2f}")

        # Detailed classification report
        target_names = [self.label_to_name[i] for i in sorted(self.label_to_name.keys())]
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, predictions, target_names=target_names))

        # Confusion matrix
        self.plot_confusion_matrix(test_labels, predictions, target_names)

        return {
            'accuracy': accuracy,
            'avg_confidence': np.mean(confidences),
            'predictions': predictions,
            'confidences': confidences
        }

    def plot_confusion_matrix(self, true_labels, predictions, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        # Save plot
        plot_path = self.data_dir / 'confusion_matrix.png'
        plt.savefig(plot_path)
        print(f"Confusion matrix saved to {plot_path}")
        plt.show()

    def save_model(self):
        """Save trained model and metadata"""
        if not self.is_trained:
            print("No trained model to save")
            return

        # Save model weights
        self.recognizer.save(str(self.model_file))

        # Save label mappings
        with open(self.labels_file, 'wb') as f:
            pickle.dump({
                'label_to_name': self.label_to_name,
                'name_to_label': self.name_to_label
            }, f)

        # Save training metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)

        print(f"Model saved to {self.model_file}")
        print(f"Labels saved to {self.labels_file}")
        print(f"Metadata saved to {self.metadata_file}")

    def load_model(self):
        """Load existing model"""
        try:
            if (self.model_file.exists() and self.labels_file.exists()
                    and self.metadata_file.exists()):
                # Load model
                self.recognizer.read(str(self.model_file))

                # Load labels
                with open(self.labels_file, 'rb') as f:
                    labels_data = pickle.load(f)
                    self.label_to_name = labels_data['label_to_name']
                    self.name_to_label = labels_data['name_to_label']

                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)

                self.is_trained = True

                print(f"Loaded {self.model_type} model with {len(self.label_to_name)} classes")
                print(f"Training info: {self.training_metadata['num_samples']} samples, "
                      f"trained on {self.training_metadata.get('timestamp', 'unknown date')}")

        except Exception as e:
            print(f"Could not load existing model: {e}")

    def predict(self, face_image):
        """Predict person from face image"""
        if not self.is_trained:
            return "No Training", 0.0

        processed_face = self.preprocess_face(face_image)
        if processed_face is None:
            return "Error", 0.0

        label, confidence = self.recognizer.predict(processed_face)

        # Convert confidence to percentage (model-specific)
        if self.model_type == 'lbph':
            # For LBPH, lower confidence is better
            confidence_pct = max(0, 100 - confidence) if confidence < 100 else 0
        else:
            # For Eigenfaces/Fisherfaces, higher confidence is better
            confidence_pct = min(100, confidence)

        person_name = self.label_to_name.get(label, "Unknown")

        return person_name, confidence_pct

    def run_live_recognition(self):
        """Run live face recognition"""
        if not self.is_trained:
            print("Model not trained. Train the model first.")
            return

        print("Starting live recognition...")
        print("Press 'q' to quit, 's' to save current frame")

        cap = cv2.VideoCapture(0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                for (x, y, w, h) in faces:
                    face_img = gray[y:y + h, x:x + w]
                    name, confidence = self.predict(face_img)

                    # Draw results
                    color = (0, 255, 0) if name != "Unknown" and confidence > 50 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    label = f"{name} ({confidence:.1f}%)"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # Just draw detection boxes
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Show model info
            info_text = f"Model: {self.model_type.upper()} | Classes: {len(self.label_to_name)}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Trainable Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'frame_{timestamp}.jpg', frame)
                print(f"Saved frame_{timestamp}.jpg")

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Trainable Face Recognition System')
    parser.add_argument('--model', choices=['lbph', 'eigenfaces', 'fisherfaces'],
                        default='lbph', help='Model type to use')
    parser.add_argument('--data-dir', default='training_data', help='Data directory')
    parser.add_argument('--dataset', help='Path to training dataset')
    parser.add_argument('--structure', choices=['person_folders', 'flat_with_labels', 'csv_labeled'],
                        default='person_folders', help='Dataset structure type')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', help='Evaluate model on test dataset')
    parser.add_argument('--live', action='store_true', help='Run live recognition')

    args = parser.parse_args()

    # Initialize recognizer
    recognizer = TrainableFaceRecognizer(
        model_type=args.model,
        data_dir=args.data_dir
    )

    if args.train:
        if not args.dataset:
            print("Error: --dataset required for training")
            return

        recognizer.train_model(
            dataset_path=args.dataset,
            structure_type=args.structure,
            validation_split=args.validation_split
        )

    if args.evaluate:
        # Load test dataset and evaluate
        test_data = recognizer.load_dataset_from_directory(args.evaluate, args.structure)
        data = recognizer.prepare_training_data(test_data, validation_split=0)
        test_faces, test_labels, _ = data['train']
        recognizer.evaluate_model(test_faces, test_labels, "Test Dataset")

    if args.live:
        recognizer.run_live_recognition()

    if not any([args.train, args.evaluate, args.live]):
        # Interactive mode
        print("Trainable Face Recognition System")
        print("=" * 50)

        while True:
            print("\nOptions:")
            print("1. Train model from dataset")
            print("2. Run live recognition")
            print("3. Evaluate model")
            print("4. Show model info")
            print("5. Exit")

            choice = input("Choose option (1-5): ").strip()

            if choice == '1':
                dataset_path = input("Enter dataset path: ").strip()
                if dataset_path and os.path.exists(dataset_path):
                    print("Dataset structures:")
                    print("1. Person folders (dataset/person1/img1.jpg)")
                    print("2. Flat with labels (dataset/person1_img1.jpg)")
                    print("3. CSV labeled (dataset/images/ + labels.csv)")

                    struct_choice = input("Choose structure (1-3): ").strip()
                    structures = {'1': 'person_folders', '2': 'flat_with_labels', '3': 'csv_labeled'}
                    structure = structures.get(struct_choice, 'person_folders')

                    recognizer.train_model(dataset_path, structure)
                else:
                    print("Invalid dataset path")

            elif choice == '2':
                recognizer.run_live_recognition()

            elif choice == '3':
                dataset_path = input("Enter test dataset path: ").strip()
                if dataset_path and os.path.exists(dataset_path):
                    test_data = recognizer.load_dataset_from_directory(dataset_path)
                    data = recognizer.prepare_training_data(test_data, validation_split=0)
                    test_faces, test_labels, _ = data['train']
                    recognizer.evaluate_model(test_faces, test_labels)
                else:
                    print("Invalid dataset path")

            elif choice == '4':
                if recognizer.is_trained:
                    print(f"Model Type: {recognizer.model_type}")
                    print(f"Classes: {len(recognizer.label_to_name)}")
                    print(f"People: {list(recognizer.label_to_name.values())}")
                    if recognizer.training_metadata:
                        print(f"Training samples: {recognizer.training_metadata.get('num_samples', 'Unknown')}")
                        print(f"Training time: {recognizer.training_metadata.get('training_time', 'Unknown'):.2f}s")
                        print(f"Trained on: {recognizer.training_metadata.get('timestamp', 'Unknown')}")
                else:
                    print("No trained model found")

            elif choice == '5':
                print("Goodbye!")
                break

            else:
                print("Invalid choice")


if __name__ == "__main__":
    main()