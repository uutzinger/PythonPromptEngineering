import cv2 as cv
import threading
from queue import Queue
import numpy as np
import time
from yolox import YoloX  # Import the YoloX class

# Queue for thread-safe frame sharing
frame_queue = Queue(maxsize=10)
running = True
fps = 0

# Initialize YOLOX Model
model_path = "object_detection_yolox_2022nov.onnx"  # Replace with your model path
yolox_model = YoloX(
    modelPath=model_path,
    confThreshold=0.5,
    nmsThreshold=0.4,
    objThreshold=0.5,
    backendId=cv.dnn.DNN_BACKEND_OPENCV,
    targetId=cv.dnn.DNN_TARGET_CPU
)

# Pre-defined classes from COCO dataset
classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def capture_images():
    """Capture images and store them in a FIFO buffer."""
    global running
    cap = cv.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Remove the oldest frame to make space
            frame_queue.put(frame)  # Add the new frame to the queue
        else:
            print("Failed to grab frame.")
    cap.release()

def visualize(frame, detections, letterbox_scale, fps=None):
    """Visualize detections with bounding boxes and labels."""
    output = frame.copy()

    if fps:
        cv.putText(output, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for det in detections:
        box = det[:4] / letterbox_scale
        box = box.astype(np.int32)
        conf = det[-2]
        class_id = int(det[-1])

        # Draw bounding box
        x0, y0, x1, y1 = box
        cv.rectangle(output, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Label
        label = f"{classes[class_id]}: {conf:.2f}"
        cv.putText(output, label, (x0, y0 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output

def display_images():
    """Retrieve images from the buffer, detect objects, and display them."""
    global running, fps
    frame_count = 0
    start_time = time.time()

    while running:
        if not frame_queue.empty():
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Get frame
            frame = frame_queue.get()

            # Preprocess the frame for YOLOX
            input_blob, letterbox_scale = letterbox(frame, target_size=(640, 640))

            # Perform inference
            detections = yolox_model.infer(input_blob)

            # Visualize results
            frame = visualize(frame, detections, letterbox_scale, fps=fps)

            # Display the frame
            cv.imshow("YOLOX Object Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv.destroyAllWindows()

def letterbox(srcimg, target_size=(640, 640)):
    """Resize and pad the input image to match target size while keeping aspect ratio."""
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

    return padded_img, ratio

# Start threads for capturing and displaying images
capture_thread = threading.Thread(target=capture_images, daemon=True)
capture_thread.start()

try:
    display_images()
except KeyboardInterrupt:
    print("Exiting...")
    running = False

capture_thread.join()
