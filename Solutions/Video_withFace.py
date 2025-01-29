import cv2 as cv
import threading
from queue import Queue
import time
import numpy as np
from yunet import YuNet  # Import YuNet class

# Queue for thread-safe frame sharing
frame_queue = Queue(maxsize=10)
running = True
fps = 0

# Initialize YuNet model
model = YuNet(
    modelPath="face_detection_yunet_2023mar.onnx",
    inputSize=[320, 320],
    confThreshold=0.6,
    nmsThreshold=0.3,
    topK=5000,
    backendId=cv.dnn.DNN_BACKEND_OPENCV,
    targetId=cv.dnn.DNN_TARGET_CPU
)

def capture_images():
    """Capture images and store them in a FIFO buffer."""
    global running
    cap = cv.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        running = False
        return

    # Set input size for YuNet based on the camera resolution
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    model.setInputSize([w, h])

    while running:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame to make space
            frame_queue.put(frame)  # Add new frame to the queue
        time.sleep(0.03)  # Simulate ~30 FPS capture rate

    cap.release()

def display_images():
    """Retrieve images from the buffer, apply YuNet, and display them."""
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

            # Get the frame
            frame = frame_queue.get()

            # Perform inference
            results = model.infer(frame)

            # Visualize results
            frame = visualize(frame, results, fps=fps)

            # Display the frame
            cv.imshow("YuNet Face Detection with FIFO", frame)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv.destroyAllWindows()

def visualize(image, results, fps=None):
    """Draw bounding boxes, landmarks, and FPS on the image."""
    output = image.copy()
    landmark_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]

    if fps:
        cv.putText(output, f"FPS: {fps:.2f}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for det in results:
        # Bounding box
        x, y, w, h = det[0:4].astype(np.int32)
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Confidence score
        conf = det[-1]
        cv.putText(output, f"{conf:.2f}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Landmarks
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, lm in enumerate(landmarks):
            cv.circle(output, tuple(lm), 2, landmark_colors[idx], -1)

    return output

# Start threads
capture_thread = threading.Thread(target=capture_images, daemon=True)
capture_thread.start()

try:
    display_images()
except KeyboardInterrupt:
    print("Exiting...")
    running = False

capture_thread.join()
