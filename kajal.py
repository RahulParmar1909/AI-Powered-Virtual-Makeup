import cv2
import numpy as np
import streamlit as st
from utils import *  # Ensure the utils file is in the same directory

# Define the mapping for occasion and skin tone (simplified for kajal)
occasion_skin_tone_mapping = {
    "Wedding": {
        "Fair/Light": {"kajal": "#000000"},  # Black kajal
        "Medium/Olive": {"kajal": "#000000"},  # Black kajal
        "Tan": {"kajal": "#000000"},  # Black kajal
        "Dark/Deep": {"kajal": "#000000"},  # Black kajal
    },
    "Birthday Party": {
        "Fair/Light": {"kajal": "#000000"},  # Black kajal
        "Medium/Olive": {"kajal": "#000000"},  # Black kajal
        "Tan": {"kajal": "#000000"},  # Black kajal
        "Dark/Deep": {"kajal": "#000000"},  # Black kajal
    },
    # Add similar mapping for other occasions...
}

# Streamlit setup
st.title("Real-Time Kajal Effect with Facial Features")

# Step 1: Ask for the occasion
occasion = st.selectbox(
    "What is the occasion?",
    ["Wedding", "Birthday Party", "Dinner Date", "Cocktail Party", "Job Interview", "New Year's Eve", "Valentine's Day", "Christmas Party", "Office Party", "Girls' Night Out"]
)

# Step 2: Open webcam for skin tone detection
video_capture = cv2.VideoCapture(0)

# Create a placeholder for displaying the frame
frame_placeholder = st.empty()

# Detect skin tone based on user’s face (simple implementation for demonstration)
def detect_skin_tone(face_landmarks):
    # Here, we will just assume the skin tone for the demo purpose.
    # You can improve this with actual color detection based on facial regions
    # For now, it returns a random skin tone for the demo.
    return "Fair/Light"  # For simplicity, returning a static skin tone

# Step 3: Define function to get kajal color based on skin tone and occasion
def get_kajal(occasion, skin_tone):
    mapping = occasion_skin_tone_mapping.get(occasion, {})
    return mapping.get(skin_tone, {"kajal": "#000000"})  # Default to black kajal

# Open the webcam stream
while True:
    success, image = video_capture.read()
    if not success:
        st.error("Error: Failed to capture image.")
        break

    image = cv2.flip(image, 1)  # Flip the image for a mirror view

    # Extract facial landmarks
    face_landmarks = read_landmarks(image=image)

    # Detect skin tone (this is where you would implement skin tone detection logic)
    skin_tone = detect_skin_tone(face_landmarks)

    # Get kajal color based on selected occasion and detected skin tone
    kajal_info = get_kajal(occasion, skin_tone)
    kajal_color = kajal_info["kajal"]

    # Convert hex color to BGR
    kajal_color_bgr = hex_to_bgr(kajal_color)

    # Focus on the eyes
    face_elements = ["LEFT_EYE", "RIGHT_EYE"]
    colors_map = {
        "LEFT_EYE": kajal_color_bgr,  # User-selected kajal color for left eye
        "RIGHT_EYE": kajal_color_bgr,  # User-selected kajal color for right eye
    }

    # Face connections based on eye elements
    face_connections = [face_points[idx] for idx in face_elements]
    colors = [colors_map[idx] for idx in face_elements]

    # Create an empty mask (same size as the image)
    mask = np.zeros_like(image)

    # Create a mask for facial features with color (only eyes)
    mask = add_mask(mask, idx_to_coordinates=face_landmarks, face_connections=face_connections, colors=colors)

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
