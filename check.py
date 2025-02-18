import dlib
import cv2

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Load the shape predictor model
predictor = dlib.shape_predictor('C:/Users/amigo/PycharmProjects/Capstone_1/shape_predictor_68_face_landmarks.dat')

# Initialize the webcam or load an image
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a path for an image

while True:
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Loop through each detected face
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)

        # Draw landmarks on the face
        for n in range(0, 68):  # 68 landmarks for the shape predictor
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the frame with the landmarks
    cv2.imshow('Face Landmark Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Example for extracting lips, eyes, and cheeks regions from landmarks

# Get the coordinates for each landmark region
def get_landmark_points(landmarks, indices):
    points = []
    for i in indices:
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append((x, y))
    return points

# Lip landmarks (points 48-67)
lip_indices = range(48, 68)
lips = get_landmark_points(landmarks, lip_indices)

# Left eye landmarks (points 36-41)
left_eye_indices = range(36, 42)
left_eye = get_landmark_points(landmarks, left_eye_indices)

# Right eye landmarks (points 42-47)
right_eye_indices = range(42, 48)
right_eye = get_landmark_points(landmarks, right_eye_indices)

# Jawline (points 0-16) - for cheeks and jaw region
jawline_indices = range(0, 17)
jawline = get_landmark_points(landmarks, jawline_indices)

# You can now use these points for further processing (like applying makeup)
import numpy as np


# Convert the region to a polygon and fill it with a color (e.g., lipstick)
def apply_color_to_region(frame, region_points, color):
    # Create a mask with the same size as the frame
    mask = np.zeros_like(frame)

    # Create the polygon based on the region points
    points = np.array(region_points, dtype=np.int32)

    # Fill the polygon with the color
    cv2.fillPoly(mask, [points], color)

    # Apply the mask to the original frame
    result = cv2.addWeighted(frame, 1.0, mask, 0.5, 0)

    return result


# Apply a red color (e.g., lipstick) to the lips
red_color = (0, 0, 255)  # Red color in BGR
frame = apply_color_to_region(frame, lips, red_color)

# Apply similar techniques for eyes and cheeks (by defining appropriate regions)
import cv2
import numpy as np
import dlib

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Path to your .dat file

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a path for an image


# Function to extract the landmarks points
def get_landmark_points(landmarks, indices):
    points = []
    for i in indices:
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append((x, y))
    return points


# Function to apply color to lips region
def apply_color_to_region(frame, region_points, color):
    # Create a mask with the same size as the frame
    mask = np.zeros_like(frame)

    # Create the polygon based on the region points
    points = np.array(region_points, dtype=np.int32)

    # Fill the polygon with the color
    cv2.fillPoly(mask, [points], color)

    # Apply the mask to the original frame
    result = cv2.addWeighted(frame, 1.0, mask, 0.5, 0)

    return result


# Main loop
while True:
    ret, frame = cap.read()

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Lip landmarks (points 48-67)
        lip_indices = range(48, 68)
        lips = get_landmark_points(landmarks, lip_indices)

        # Define the red color for the lips (BGR format)
        red_color = (0, 0, 255)  # Red in BGR

        # Apply red color to lips
        frame = apply_color_to_region(frame, lips, red_color)

    # Display the frame with red lips
    cv2.imshow("Virtual Makeup - Red Lips", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
