import cv2
import numpy as np
import streamlit as st
from utils import *  # Ensure the utils file is in the same directory

# Streamlit setup
st.title("Real-Time Lip Gloss Effect with Facial Features")

# User input for lipstick color and gloss effect
lipstick_color = st.color_picker("Choose Lipstick Color", "#ff69b4")  # Default to pink
lip_gloss = st.checkbox("Apply Lip Gloss Effect", value=False)

# Convert the chosen color from hex to BGR
lipstick_color_bgr = hex_to_bgr(lipstick_color)

# OpenCV video capture for webcam
video_capture = cv2.VideoCapture(0)

# Only focus on lips
face_elements = ["LIP_UPPER", "LIP_LOWER"]

# Change the color of lips (based on user selection)
colors_map = {
    "LIP_UPPER": lipstick_color_bgr,  # User-selected color for upper lip
    "LIP_LOWER": lipstick_color_bgr,  # User-selected color for lower lip
}

# Face connections based on lip elements
face_connections = [face_points[idx] for idx in face_elements]
colors = [colors_map[idx] for idx in face_elements]

# Streamlit loop to show the video stream
frame_placeholder = st.empty()  # Placeholder to display the webcam frame

while True:
    success, image = video_capture.read()
    if not success:
        st.error("Error: Failed to capture image.")
        break

    image = cv2.flip(image, 1)  # Flip the image for a mirror view

    # Create an empty mask (same size as the image)
    mask = np.zeros_like(image)

    # Extract facial landmarks
    face_landmarks = read_landmarks(image=image)

    # Create a mask for facial features with color (only lips)
    mask = add_mask(
        mask,
        idx_to_coordinates=face_landmarks,
        face_connections=face_connections,
        colors=colors
    )

    # Apply lip gloss effect if selected
    if lip_gloss:
        gloss_color = [255, 105, 180]  # Glossy pink (BGR)
        mask = add_mask(
            mask,
            idx_to_coordinates=face_landmarks,
            face_connections=face_connections,
            colors=[gloss_color] * len(face_connections)  # Apply gloss to both lips
        )

    # Combine the original image and the mask with weights to get the final output
    output = cv2.addWeighted(image, 1.0, mask, 0.2, 1.0)

    # Convert output image to RGB for Streamlit display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # Display the result using Streamlit (live updates)
    frame_placeholder.image(output_rgb, channels="RGB", use_column_width=True)

    # Break the loop if 'q' is pressed (in case of debugging)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
