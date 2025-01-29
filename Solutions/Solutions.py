import cv2

# Load the original image
image = cv2.imread('ship.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image. Please make sure the file path is correct.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the original image
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

    # Display the original image
    cv2.imshow('Original Image', image)

    # Display the grayscale image
    cv2.imshow('Grayscale Image', gray_image)

    # Display the blurred image
    cv2.imshow('Blurred Image', blurred_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Get the image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate the image by 30 degrees
    matrix_30 = cv2.getRotationMatrix2D(center, 30, 1.0)  # Rotation matrix
    rotated_30 = cv2.warpAffine(image, matrix_30, (w, h))  # Apply rotation

    # Rotate the image by 90 degrees
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Rotate the image by 180 degrees
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Rotated by 30 Degrees', rotated_30)
    cv2.imshow('Rotated by 90 Degrees', rotated_90)
    cv2.imshow('Rotated by 180 Degrees', rotated_180)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import random

# Load the original image
image = cv2.imread('cards.jpg')

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not load image. Please make sure the file path is correct.")
    exit()

# Step 1: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Step 4: Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Draw contours larger than 20 pixels on the original image
contour_image = image.copy()
large_contours_count = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 20:  # Check if the contour area is larger than 20 pixels
        large_contours_count += 1
        # Generate a random color for the contour
        color = tuple(random.choices(range(256), k=3))
        # Draw the contour on the image
        cv2.drawContours(contour_image, [contour], -1, color, 2)

# Print the number of large contours
print(f"Number of objects found with area > 20 pixels: {large_contours_count}")

# Display each step
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Edge Detection', edges)
cv2.imshow('Contours Larger Than 20 Pixels', contour_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'c' to capture an image, or 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the live feed
    cv2.imshow("Live Camera Feed", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture the image on pressing 'c'
        # Display the captured image in a new window
        cv2.imshow("Captured Image", frame)
        print("Image captured. Press any key to close the captured image.")
        cv2.waitKey(0)  # Wait for a key press in the captured image window
        cv2.destroyWindow("Captured Image")  # Close the captured image window

    elif key == ord('q'):  # Quit the program on pressing 'q'
        print("Quitting...")
        break
