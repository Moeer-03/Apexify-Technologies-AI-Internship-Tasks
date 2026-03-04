"""
Image Classifier with Google's Teachable Machine Model
========================================================
Classifies images (from webcam) using a pre-trained Teachable Machine model.

Supports:
- TensorFlow/Keras models (requires Python 3.11-3.12)
- TensorFlow Lite models (more compatible)
- Demo mode (works on any Python version for testing)

How to use Teachable Machine:
1. Go to: https://teachablemachine.withgoogle.com/
2. Create a New Project > Image Project
3. Create classes: "Happy Face", "Sad Face", "Neutral"
4. Train the model
5. Export as TensorFlow > Download
6. Place the model folder in the same directory as this script

The exported folder should contain:
- keras_model.h5 (model file) OR model.tflite (TensorFlow Lite)
- labels.txt (class labels)

NOTE: Python 3.14 and 3.13 don't yet have TensorFlow wheel support on Windows.
For production use, use Python 3.11 or 3.12 with TensorFlow installed.
This script includes demo mode for testing!
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
import json
from typing import Tuple, Dict

# Check if TensorFlow or TensorFlow Lite is available
TF_AVAILABLE = False
try:
    import tensorflow
    TF_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False


class TeachableMachineClassifier:
    """
    Image classifier using Google's Teachable Machine model.
    """

    def __init__(self, model_path="teachable_machine_model", use_demo=False):
        """
        Initialize classifier with a Teachable Machine model.

        Args:
            model_path: Path to the exported Teachable Machine model directory
            use_demo: If True, use demo mode (simulates classifier for testing)
        """
        self.model_path = model_path
        self.model = None
        self.labels = []
        self.loaded = False
        self.use_demo = use_demo

        if self.use_demo:
            print("Using DEMO MODE (without real model)")
            self.demo_mode()
            return

        if not TF_AVAILABLE:
            print("\n⚠️  TensorFlow/TensorFlow Lite not available")
            print("For Python 3.14, install TensorFlow Lite: pip install tflite-runtime")
            print("Or use demo mode for testing")
            return

        self.load_model()

    def demo_mode(self):
        """Enable demo mode for testing without a real model."""
        self.labels = ["Happy Face", "Sad Face", "Neutral"]
        self.loaded = True
        print(f"✓ Demo mode enabled")
        print(f"✓ Classes: {', '.join(self.labels)}")

    def load_model(self):
        """Load the Teachable Machine model and labels."""
        # Try loading TensorFlow Lite model first (more compatible)
        model_file_tflite = os.path.join(self.model_path, "model.tflite")
        model_file = os.path.join(self.model_path, "keras_model.h5")
        labels_file = os.path.join(self.model_path, "labels.txt")

        # Check for TensorFlow Lite model
        if os.path.exists(model_file_tflite):
            self._load_tflite_model(model_file_tflite, labels_file)
            return

        # Check for Keras model
        if os.path.exists(model_file):
            self._load_keras_model(model_file, labels_file)
            return

        print(f"\n❌ Model file not found!")
        print(f"   Looked for:")
        print(f"   - {model_file_tflite}")
        print(f"   - {model_file}")
        print("\nTo use this script:")
        print("1. Go to: https://teachablemachine.withgoogle.com/")
        print("2. Create a new Image Project")
        print("3. Create classes: 'Happy Face', 'Sad Face', 'Neutral'")
        print("4. Train your model")
        print("5. Export as TensorFlow > Download (Keras)")
        print("6. Extract the downloaded folder as 'teachable_machine_model'")
        print("7. Place it in the same directory as this script")
        print("\nOr use demo mode for testing: classifier = TeachableMachineClassifier(use_demo=True)")

    def _load_keras_model(self, model_file, labels_file):
        """Load Keras/TensorFlow model."""
        try:
            from tensorflow.keras.models import load_model
            print("Loading Keras model...")
            self.model = load_model(model_file, compile=False)
            print("✓ Model loaded!")
        except ImportError:
            print("❌ TensorFlow not available for Keras model")
            return
        except Exception as e:
            print(f"❌ Error loading Keras model: {e}")
            return

        self._load_labels(labels_file)

    def _load_tflite_model(self, model_file, labels_file):
        """Load TensorFlow Lite model."""
        try:
            import tflite_runtime.interpreter as tflite
            print("Loading TensorFlow Lite model...")
            self.model = tflite.Interpreter(model_path=model_file)
            self.model.allocate_tensors()
            print("✓ TensorFlow Lite model loaded!")
            self.loaded = True
        except ImportError:
            print("❌ TensorFlow Lite not installed")
            print("Install with: pip install tflite-runtime")
            return
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            return

        self._load_labels(labels_file)

    def _load_labels(self, labels_file):
        """Load class labels."""
        if not os.path.exists(labels_file):
            print(f"❌ Labels file not found: {labels_file}")
            return

        try:
            with open(labels_file, "r") as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✓ Classes: {', '.join(self.labels)}")
            self.loaded = True
        except Exception as e:
            print(f"❌ Error loading labels: {e}")

    def classify_frame(self, frame):
        """
        Classify an image frame.

        Args:
            frame: OpenCV frame (BGR image)

        Returns:
            Tuple of (class_name, confidence, all_predictions)
        """
        if not self.loaded or self.model is None:
            if self.use_demo:
                return self._classify_demo(frame)
            return "Model not loaded", 0.0, {}

        try:
            # Prepare image for model
            # Teachable Machine expects 224x224 images
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0  # Normalize

            # Check if it's a Keras model or TFLite model
            if hasattr(self.model, 'predict'):  # Keras model
                img_array = np.expand_dims(img_array, axis=0)
                predictions = self.model.predict(img_array, verbose=0)
                prediction_array = predictions[0]
            else:  # TFLite model
                prediction_array = self._classify_tflite(img_array)

            # Get class with highest confidence
            class_idx = np.argmax(prediction_array)
            confidence = float(prediction_array[class_idx])
            class_name = self.labels[class_idx] if class_idx < len(self.labels) else "Unknown"

            # Create dict of all predictions
            all_predictions = {
                self.labels[i]: float(prediction_array[i])
                for i in range(len(self.labels))
            }

            return class_name, confidence, all_predictions

        except Exception as e:
            print(f"❌ Error during classification: {e}")
            return "Error", 0.0, {}

    def _classify_tflite(self, img_array):
        """Classify using TensorFlow Lite model."""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        # Set input
        self.model.set_tensor(input_details[0]['index'], np.expand_dims(img_array, axis=0))
        self.model.invoke()

        # Get output
        output_data = self.model.get_tensor(output_details[0]['index'])
        return output_data[0]

    def _classify_demo(self, frame):
        """Classify in demo mode (simulates classifier)."""
        # In demo mode, classify based on image brightness as a simple example
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Simple heuristic: brightness determines class
        if brightness < 85:
            prediction_array = np.array([0.8, 0.1, 0.1])  # Mostly dark -> Sad
        elif brightness < 170:
            prediction_array = np.array([0.2, 0.2, 0.6])  # Medium -> Neutral
        else:
            prediction_array = np.array([0.7, 0.2, 0.1])  # Bright -> Happy

        class_idx = np.argmax(prediction_array)
        confidence = float(prediction_array[class_idx])
        class_name = self.labels[class_idx]

        all_predictions = {
            self.labels[i]: float(prediction_array[i])
            for i in range(len(self.labels))
        }

        return class_name, confidence, all_predictions

    def process_webcam(self):
        """Capture from webcam and classify images in real-time."""
        if not self.loaded:
            print("❌ Model not loaded. Cannot process webcam.")
            return

        print("\n" + "=" * 60)
        print("IMAGE CLASSIFIER WITH TEACHABLE MACHINE")
        print("=" * 60)
        print(f"Classes to detect: {', '.join(self.labels)}")
        print("\nStarting webcam...")
        print("Press 'q' to quit, 's' to save a frame\n")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        fps_clock = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break

                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)

                # Classify the frame
                class_name, confidence, predictions = self.classify_frame(frame)

                # Display classification results
                h, w = frame.shape[:2]

                # Background for text
                cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
                cv2.rectangle(frame, (0, 0), (w, 120), (0, 255, 0), 2)

                # Main classification
                text_main = f"{class_name}: {confidence:.1%}"
                cv2.putText(
                    frame,
                    text_main,
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

                # All predictions
                y_offset = 65
                for label, conf in predictions.items():
                    text = f"{label}: {conf:.1%}"
                    color = (0, 255, 0) if label == class_name else (200, 200, 200)
                    cv2.putText(
                        frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        1,
                    )
                    y_offset += 25

                # FPS counter
                frame_count += 1
                elapsed = time.time() - fps_clock
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    fps_clock = time.time()
                    frame_count = 0
                else:
                    fps = frame_count / elapsed

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Display
                cv2.imshow("Teachable Machine Classifier", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n✓ Exiting...")
                    break
                elif key == ord("s"):
                    filename = f"classification_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved: {filename}")

        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def classify_image(self, image_path):
        """
        Classify a single image file.

        Args:
            image_path: Path to image file
        """
        if not self.loaded:
            print("❌ Model not loaded")
            return

        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return

        print(f"\nClassifying: {image_path}")
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"❌ Cannot read image: {image_path}")
            return

        class_name, confidence, predictions = self.classify_frame(frame)

        print(f"\nResults:")
        print(f"  Primary: {class_name} ({confidence:.1%})")
        print(f"  All predictions:")
        for label, conf in predictions.items():
            print(f"    - {label}: {conf:.1%}")


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("TEACHABLE MACHINE IMAGE CLASSIFIER")
    print("=" * 70)

    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"\nPython Version: {python_version}")

    # Determine best mode based on Python version and available packages
    use_demo = True
    tf_available = False

    try:
        import tensorflow
        tf_available = True
        use_demo = False
    except ImportError:
        try:
            import tflite_runtime
            tf_available = True
            use_demo = False
        except ImportError:
            use_demo = True

    if use_demo:
        print(f"Status: Python {python_version} - Using DEMO MODE")
        print("         (TensorFlow not available on this Python version)")
    else:
        print(f"Status: TensorFlow available - Can use real models")

    print("\nOptions:")
    print("1. Classify from webcam (real-time)")
    print("2. Classify image file")
    print("3. View instructions for creating/using real models")
    print("4. Exit")

    # Initialize classifier
    classifier = TeachableMachineClassifier(use_demo=use_demo)

    if not classifier.loaded:
        print("\n❌ Classifier not initialized. Exiting...")
        return

    print()

    while True:
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            classifier.process_webcam()
        elif choice == "2":
            image_path = input("Enter image file path: ").strip()
            classifier.classify_image(image_path)
        elif choice == "3":
            show_instructions()
        elif choice == "4":
            print("✓ Exiting...")
            break
        else:
            print("❌ Invalid choice. Please try again.")


def show_instructions():
    """Show instructions for creating and using Teachable Machine models."""
    print("\n" + "=" * 70)
    print("INSTRUCTIONS: Creating a Teachable Machine Model")
    print("=" * 70)
    print("""
STEP 1: GO TO TEACHABLE MACHINE
   Visit: https://teachablemachine.withgoogle.com/
   
STEP 2: CREATE PROJECT
   - Click "Create Project"
   - Select "Image Project"
   
STEP 3: CREATE CLASSES
   - Class 1: "Happy Face" (add images of happy faces)
   - Class 2: "Sad Face" (add images of sad faces)
   - Class 3: "Neutral" (add images of neutral faces)
   - Each class needs 5-10 sample images
   
STEP 4: TRAIN THE MODEL
   - Click "Train Model" (happens in browser)
   - Wait for training to complete
   
STEP 5: EXPORT THE MODEL
   - Click "Export"
   - Choose "TensorFlow"
   - Click "Download" (Gets a .zip file)
   
STEP 6: PREPARE FOR THIS SCRIPT
   - Extract the downloaded .zip file
   - Rename the extracted folder to: "teachable_machine_model"
   - Place it in the same directory as this script (task3.py)
   
STEP 7: USE WITH PYTHON 3.11 OR 3.12
   - Create virtual environment: py -3.12 -m venv tf_env
   - Activate it: tf_env\\Scripts\\Activate.ps1
   - Install TensorFlow: pip install tensorflow opencv-python
   - Run this script: python task3.py
   
STEP 8: SELECT REAL MODEL MODE
   - When prompted, the script will auto-detect and use your model!
   
NOTES:
   - Python 3.14/3.13: Use DEMO MODE (no TensorFlow available yet)
   - Python 3.12/3.11: Full support with real models
   - Teachable Machine automatically handles image preprocessing
   - Model works offline once downloaded
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
