# Prompt Engineering
We will create computer code with AI.



- [Prompt Engineering](#prompt-engineering)
  - [Image and Video Processing](#image-and-video-processing)
    - [Load and Manipulate Images](#load-and-manipulate-images)
    - [Count Objects](#count-objects)
    - [Object Detection](#object-detection)
  - [Audio Processing](#audio-processing)
  - [Reading and Processing Data Files](#reading-and-processing-data-files)

Created by Urs Utzinger January 2025 with inspiration from [Python for Engineers](https://new.pythonforengineers.com)

Solutions are included but the goal is to create them by yourself.

## Image and Video Processing
### Load and Manipulate Images

1) Ask your AI agent to load the image.jpg and display it using opencv.

For example:
Prompt: ```In python using opencv load the image ship.jpg and display it```
Answer:
```
import cv2

# Load the image
image = cv2.imread('ship.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image. Please make sure the file path is correct.")
else:
    # Display the image
    cv2.imshow('Ship Image', image)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

2) Ask your AI agent to convert the image to gray scale after loading it and to display both the original and gray scale image.

3) Ask your AI agent to blur the grayscale image and display it.

 <a href="./Ship_Images.png" target="_blank"> <img src="./Ship_Images.png" width="600"> </a> 


4) Ask your AI agent to apply an edge detection algorithm to the blurred image and display it.

5) Ask your AI agent to rotate the image 90 degrees, 180 degrees and 30 degrees and display them.

6) Ask your AI agent what an affine transformation is what different types of transformations there are besides rotation.

<a href="./Ship_Images_Rotated.png" target="_blank"> <img src="./Ship_Images_Rotated.png" width="600"> </a>

### Count Objects

1) Ask your AI agent to load the cards.jpg image and display it.

2) Now ask it to blur it and then perform edge detection with the purpose to find contours in the edges. Ask it to display each contour on top of the original image.

3) Ask your agent how contour detection works.

4) Ask it to modify the program so that it lets you know how many contours it found that enclose more than 50 pixels and the outline each found object with different color.

5) Ask it if there are other methods one could use to separate the contours, for example based on shape.

<a href="./Example_Cards.png" target="_blank"> <img src="./Example_Cards.png" width="600"> </a>

### Object Detection

1) Ask your AI agent to create a python program that opens the camera, obtains an image and displays it. Specify that you want to obtain each image separately before displaying it.

2) Ask your AI agent to continuously grab image and put them into a first in - first out buffer (FIFO). In a separate thread you will want to grab an image and display it. Ask that it creates a frames per second measurement and displaye the number to the bottom left of the image.

Mine looks like that:

```python
import cv2
import threading
from queue import Queue
import time

# Queue for thread-safe frame sharing
frame_queue = Queue(maxsize=10)
running = True
fps = 0

def capture_images():
    """Capture images in a thread-safe way and put them in a queue."""
    global running
    cap = cv2.VideoCapture(0)  # Default camera index
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame to make space
            frame_queue.put(frame)  # Add the new frame to the queue
        time.sleep(0.03)  # Simulate ~30 FPS capture rate

    cap.release()

def display_images():
    """Retrieve images from the queue and display them with FPS."""
    global running, fps
    frame_count = 0
    start_time = time.time()

    while running:
        if not frame_queue.empty():
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Update FPS every second
                fps = frame_count
                frame_count = 0
                start_time = time.time()

            # Get the frame and draw the FPS
            frame = frame_queue.get()
            cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Camera Output", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()

# Create and start threads
capture_thread = threading.Thread(target=capture_images, daemon=True)
capture_thread.start()

try:
    display_images()
except KeyboardInterrupt:
    print("Exiting...")
    running = False

capture_thread.join()

```

The advantage of this code is that you run basically two programs. The one that captures the images and one that analyzes and displays them. If you were to run this sequentially you would not be wasting the time when the image is transferring to your computer. This way images arrive contiguously. 

<a href="./Camera_ok.png" target="_blank"> <img src="./Camera_ok.png" width="300"> </a>


3) Ask it to help you obtain neural network models from https://github.com/opencv/opencv_zoo.  You want a copy on your computer.

4) Ask it to use the previous program to capture video and to include code to load the face detection model and apply it to each frame and then to draw the bounding box and land marks on the image. You will need to find the appropriate model or ask the AI agent to search the answer on the model zoo website. Since these models are fairly new the AI agent likely does not know exactly how to use them. Once you know the model you can find the example code from the website (e.g. demo.py). You can either upload the example code or paste it together with your question. I asked it to look at the code and include it to provide the solution. Your solution does not need to look like mine.

<a href="./Camera_Face.png" target="_blank"> <img src="./Camera_Face.png" width="600"> </a>

You can see that it found my pupils, nose and edge of mouth.

5) Ask the AI agent what good model would be for hand and finger detection from the same model zoo. Then find it in the downloaded opencv_zoo.

6) Use the demo and other python program and ask the AI agent to incorporate in the previous program replacing the face detection section.

<a href="./Hands.png" target="_blank"> <img src="./Hands.png" width="600"> </a>

This is very interesting as it now shows all the hand and finger segments. You could decode it for sign language or do some simple gesture analysis but we dont do that now.

7) In some of the lecture notes the instructor shows YOLO on a James Bond movie. The opencv zoo has YOLOX. Ask that your program recognizes simple objects such as the ones in a YOLO algorithm. You likely will need to give the AI agent the example code. You can just include it with your question. I have the example programs in in `opencv_zoo/models/object_detection_yolox`.

<a href="./Example_Yolox.png" target="_blank"> <img src="./Example_Yolox.png" width="600"> </a>

Its working quite well but is slow.It thinks my wallet is a phone. I bet if opencv would use the cuda extension (NVIDEA graphics cards) it would be faster. But for that you need to build/compile opencv and it will take quite some time. So lets not do that.

## Audio Processing

1) Ask you AI agent to create a sin wave at 1000Hz, display it in a graph with matlplot lib and play it over the speakers of your computer. It's possible you need to tell it what computer you are using so that it can use the appropriate library for audio playback.

<a href="./Example_Sinewave.png" target="_blank"> <img src="./Example_Sinewave.png" width="600"> </a>

2) Ask it to plot the power spectrum of your signal and to use logarithmic scale on the y axis.

<a href="./Example_Power.png" target="_blank"> <img src="./Example_Power.png" width="600"> </a>

I like these plots and they look very similar to what Matlab does.

3) Often we have noise from the power lines in the audio. That noise is at 60Hz. You can ask your AI agent to add a noise sinewave at 60Hz and make its amplitude 1/3 of the main frequency.

<a href="./Example_NoiseandSine.png" target="_blank"> <img src="./Example_NoiseandSine.png" width="600"> </a>

<a href="./Example_PowerNoise.png" target="_blank"> <img src="./Example_PowerNoise.png" width="600"> </a>

This looks as expected.

4) Now we want to create a filer on the audio signal that suppresses the signal at 60Hz but leaves the other signal intact. Often this is called a notch filter at 60Hz. You will need it in BME417 if you take that class.

<a href="./time_domain_plot_filter.png" target="_blank"> <img src="./time_domain_plot_filter.png" width="600"> </a>
<a href="./power_spectrum_filter.png" target="_blank"> <img src="./power_spectrum_filter.png" width="600"> </a>

5) Lets try this in realtime. That can be difficult with python because often its not used as realtime processing tool. ChatGPT suggests that I use pyqtgraph to display the signal and the spectrum. I want to record the audio in realtime, filter it and display the signal as well as the filtered power spectrum. Since I use pyqtgraph it likely will also want to use pyqt. The AI agents should be able to tell you want you need to install.

I changed the notch frequency to 2000Hz and I whistle. I can observe that the power spectrum is reduced when i hit that frequency. You can create the tone with an other python program which you can run simultaneously. Visual Studio Code will ask you if you really want to run two program.:

```python
import numpy as np
import sounddevice as sd

fs = 44100  # Sampling rate (Hz)
duration = 2.0  # seconds
frequency = 1000  # Hz (tone frequency)

t = np.linspace(0, duration, int(fs * duration), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate sine wave

sd.play(tone, samplerate=fs)
sd.wait()
```

<a href="./Audio_realtime_1000.png" target="_blank"> <img src="./Audio_realtime_1000.png" width="600"> </a>

<a href="./Audio_realtime_2000.png" target="_blank"> <img src="./Audio_realtime_2000.png" width="600"> </a>


## Reading and Processing Data Files

1) We want to analyze obesity data. As example we use data from England: Statistics on Obesity, Physical Activity and Diet 2020: Data tables





