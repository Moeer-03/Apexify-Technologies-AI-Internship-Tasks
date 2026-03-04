import cv2
import numpy as np
import os
from collections import defaultdict, deque
from pathlib import Path
import time

# Try to import YOLOv8 (recommended)
try:
    from ultralytics import YOLO
    YOLO_V8_AVAILABLE = True
except ImportError:
    YOLO_V8_AVAILABLE = False
    print("Note: YOLOv8 not installed. Install with: pip install ultralytics")


class CentroidTracker:
    """
    Simple object tracker using centroid distance.
    Tracks objects by finding the closest centroid between frames.
    """

    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize tracker.

        Args:
            max_disappeared: Maximum frames object can disappear before removal
            max_distance: Maximum distance between centroids to consider same object
        """
        self.next_object_id = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.track_length = 50
        self.tracks = defaultdict(lambda: deque(maxlen=self.track_length))

    def register(self, centroid):
        """Register a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister an object."""
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.tracks.pop(object_id, None)

    def update(self, rects):
        """
        Update tracker with detections.

        Args:
            rects: List of bounding boxes [(x1, y1, x2, y2), ...]

        Returns:
            Dictionary of {object_id: centroid}
        """
        if len(rects) == 0:
            # No detections, increment disappeared count
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate centroids of current detections
        input_centroids = np.zeros((len(rects), 2))
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids[i] = [cx, cy]

        # If no objects being tracked, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match detections to existing objects using distance
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Calculate distances between all pairs
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dist = np.linalg.norm(object_centroids[i] - input_centroids[j])
                    distances[i][j] = dist

            # Find minimum distance matches
            used_rows = set()
            used_cols = set()

            for _ in range(min(len(object_centroids), len(input_centroids))):
                object_idx = distances[~np.isin(np.arange(len(object_centroids)), list(used_rows))].argmin() // len(input_centroids)
                input_idx = distances[~np.isin(np.arange(len(object_centroids)), list(used_rows))].argmin() % len(input_centroids)

                actual_object_idx = np.where(~np.isin(np.arange(len(object_centroids)), list(used_rows)))[0][object_idx]
                actual_input_idx = input_idx

                if distances[actual_object_idx, actual_input_idx] < self.max_distance:
                    object_id = object_ids[actual_object_idx]
                    self.objects[object_id] = input_centroids[actual_input_idx]
                    self.disappeared[object_id] = 0
                    self.tracks[object_id].append(input_centroids[actual_input_idx])
                    used_rows.add(actual_object_idx)
                    used_cols.add(actual_input_idx)

            # Register new detections
            unused_input_indices = np.where(~np.isin(np.arange(len(input_centroids)), list(used_cols)))[0]
            for idx in unused_input_indices:
                self.register(input_centroids[idx])

            # Deregister disappeared objects
            for i, object_id in enumerate(object_ids):
                if i not in used_rows:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

        return self.objects


class ObjectDetectionTracker:
    """Real-time object detection and tracking using YOLO."""

    def __init__(self, model_name="yolov8n", confidence=0.5, use_yolov8=True):
        """
        Initialize detector and tracker.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, etc.)
            confidence: Confidence threshold for detections
            use_yolov8: Use YOLOv8 if available
        """
        self.confidence = confidence
        self.model = None
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=50)
        self.class_names = []
        self.colors = {}  # color for each class

        if use_yolov8 and YOLO_V8_AVAILABLE:
            self.load_yolov8(model_name)
        else:
            self.load_yolov3()

    def load_yolov8(self, model_name="yolov8n"):
        """Load YOLOv8 model."""
        try:
            print(f"Loading YOLOv8 model: {model_name}")
            self.model = YOLO(f"{model_name}.pt")
            self.class_names = self.model.names.values()
            print(f"✓ YOLOv8 model loaded! ({len(self.class_names)} classes)")
            return True
        except Exception as e:
            print(f"Error loading YOLOv8: {e}")
            return False

    def load_yolov3(self):
        """Load YOLOv3 model (fallback)."""
        try:
            print("Downloading YOLOv3 model files...")

            config_path = "yolov3.cfg"
            weights_path = "yolov3.weights"
            names_path = "coco.names"

            # Download files if not present
            if not os.path.exists(weights_path):
                print("Downloading YOLOv3 weights (~236MB)...")
                import urllib.request
                url = "https://pjreddie.com/media/files/yolov3.weights"
                urllib.request.urlretrieve(url, weights_path)

            if not os.path.exists(config_path):
                url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
                import urllib.request
                urllib.request.urlretrieve(url, config_path)

            if not os.path.exists(names_path):
                url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
                import urllib.request
                urllib.request.urlretrieve(url, names_path)

            # Load network
            self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            # Load class names
            with open(names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]

            print(f"✓ YOLOv3 model loaded! ({len(self.class_names)} classes)")
            return True

        except Exception as e:
            print(f"Error loading YOLOv3: {e}")
            return False

    def detect_objects(self, frame):
        """
        Detect objects in frame.

        Returns:
            List of detections: [(x1, y1, x2, y2, class_name, confidence), ...]
        """
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        if YOLO_V8_AVAILABLE and hasattr(self.model, 'predict'):
            # YOLOv8 detection
            results = self.model.predict(frame, conf=self.confidence, verbose=False)

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]

                    detections.append((x1, y1, x2, y2, class_name, conf))

            return detections
        else:
            # YOLOv3 detection (legacy)
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
            self.model.setInput(blob)

            layer_names = self.model.getUnconnectedOutLayersNames()
            outputs = self.model.forward(layer_names)

            detections = []
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence:
                        center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                        width, height = int(detection[2] * w), int(detection[3] * h)

                        x1 = center_x - width // 2
                        y1 = center_y - height // 2
                        x2 = x1 + width
                        y2 = y1 + height

                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                        detections.append((x1, y1, x2, y2, class_name, float(confidence)))

            return detections

    def get_color(self, class_name):
        """Get consistent color for class."""
        if class_name not in self.colors:
            self.colors[class_name] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[class_name]

    def process_frame(self, frame):
        """
        Process a frame: detect objects and track them.

        Returns:
            Processed frame with detections and tracking IDs
        """
        # Detect objects
        detections = self.detect_objects(frame)

        # Extract bounding boxes for tracking
        rects = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in detections]

        # Update tracker
        tracked_objects = self.tracker.update(rects)

        # Create a copy for drawing
        output_frame = frame.copy()

        # Draw tracked objects
        for object_id, centroid in tracked_objects.items():
            # Find corresponding detection
            if object_id < len(detections):
                x1, y1, x2, y2, class_name, confidence = detections[object_id]
                color = self.get_color(class_name)

                # Draw bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label with tracking ID
                label = f"ID:{object_id} {class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_bg = (x1, max(y1 - label_size[1] - 5, 0), x1 + label_size[0] + 5, y1)
                cv2.rectangle(output_frame, label_bg[:2], label_bg[2:], color, -1)
                cv2.putText(output_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw centroid
                cv2.circle(output_frame, tuple(map(int, centroid)), 4, color, -1)

                # Draw tracking line
                if object_id in self.tracker.tracks:
                    track = self.tracker.tracks[object_id]
                    for i in range(1, len(track)):
                        pt1 = tuple(map(int, track[i - 1]))
                        pt2 = tuple(map(int, track[i]))
                        cv2.line(output_frame, pt1, pt2, color, 1)

        return output_frame, len(tracked_objects)

    def process_video(self, source=0, output_path=None):
        """
        Process video from webcam or file.

        Args:
            source: 0 for webcam, or path to video file
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        print(f"\nProcessing video...")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames if total_frames > 0 else 'Unknown'}")
        print(f"Press 'q' to quit, 's' to save frame\n")

        frame_count = 0
        start_time = time.time()
        fps_counter = deque(maxlen=30)

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Process frame
                output_frame, num_objects = self.process_frame(frame)

                # Calculate FPS
                elapsed = time.time() - start_time
                fps_counter.append(1.0 / (elapsed - (frame_count - 1) / 30) if elapsed > 0 else 0)
                current_fps = np.mean(list(fps_counter)) if fps_counter else 0

                # Display stats
                stats_text = f"Frame: {frame_count} | Objects: {num_objects} | FPS: {current_fps:.1f}"
                cv2.putText(output_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Object Detection and Tracking', output_frame)

                # Write frame
                if out:
                    out.write(output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"detection_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, output_frame)
                    print(f"Frame saved: {filename}")

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete! Total frames: {frame_count}")


def main():
    """Main application."""
    print("\n" + "="*70)
    print("OBJECT DETECTION AND TRACKING")
    print("="*70)

    if not YOLO_V8_AVAILABLE:
        print("\n⚠ YOLOv8 not installed. Install with:")
        print("  pip install ultralytics")
        print("\nFalling back to YOLOv3 (requires downloading ~236MB)...\n")

    # Initialize detector
    detector = ObjectDetectionTracker(
        confidence=0.5,
        use_yolov8=YOLO_V8_AVAILABLE
    )

    if detector.model is None:
        print("Error: Could not load any detection model")
        return

    while True:
        print("\n" + "-"*70)
        print("Select input source:")
        print("1. Webcam (real-time)")
        print("2. Video file")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            print("\nStarting webcam detection and tracking...")
            detector.process_video(source=0)

        elif choice == "2":
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                output_path = input("Save output video? (leave blank to skip): ").strip()
                detector.process_video(source=video_path, output_path=output_path if output_path else None)
            else:
                print(f"Error: File not found - {video_path}")

        elif choice == "3":
            print("\nThank you for using Object Detection and Tracking!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
