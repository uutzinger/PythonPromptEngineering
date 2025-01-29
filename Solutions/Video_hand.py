import cv2 as cv
import threading
from queue import Queue
import time
import numpy as np
from mp_handpose import MPHandPose  # Import the MediaPipe HandPose class
from mp_palmdet import MPPalmDet  # Import the Palm Detection class

# Queue for thread-safe frame sharing
frame_queue = Queue(maxsize=10)
running = True

# Initialize Palm Detector and HandPose model
palm_model_path = "palm_detection_mediapipe_2023feb.onnx"  # Replace with your actual path
hand_model_path = "handpose_estimation_mediapipe_2023feb.onnx"  # Replace with your actual path
palm_detector = MPPalmDet(modelPath=palm_model_path, scoreThreshold=0.6, nmsThreshold=0.3)
handpose_detector = MPHandPose(modelPath=hand_model_path, confThreshold=0.8)

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
        time.sleep(0.03)  # Simulate ~30 FPS capture rate

    cap.release()

def visualize(image, hands):
    """Draw detected hands and landmarks on the image."""
    output = image.copy()

    for handpose in hands:
        conf = handpose[-1]
        bbox = handpose[0:4].astype(np.int32)
        landmarks = handpose[4:67].reshape(21, 3).astype(np.int32)

        # Draw bounding box
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Draw landmarks
        for lm in landmarks:
            cv.circle(output, (lm[0], lm[1]), 5, (0, 0, 255), -1)

    return output

def display_images():
    """Retrieve images from the buffer, detect hands, and display them."""
    global running

    while running:
        if not frame_queue.empty():
            # Get the frame from the queue
            frame = frame_queue.get()

            # Detect palms
            palms = palm_detector.infer(frame)
            hands = np.empty((0, 132))  # Placeholder for hand detection results

            # Detect hand poses for each detected palm
            for palm in palms:
                handpose = handpose_detector.infer(frame, palm)
                if handpose is not None:
                    hands = np.vstack((hands, handpose))

            # Visualize results
            frame = visualize(frame, hands)

            # Display the frame
            cv.imshow("Hand Detection with MediaPipe", frame)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv.destroyAllWindows()

# Start threads for capturing and displaying images
capture_thread = threading.Thread(target=capture_images, daemon=True)
capture_thread.start()

try:
    display_images()
except KeyboardInterrupt:
    print("Exiting...")
    running = False

capture_thread.join()
