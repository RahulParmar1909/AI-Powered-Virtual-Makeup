import cv2
import numpy as np
import streamlit as st
from utils import *  # Ensure the utils file is in the same directory

# Define the mapping for occasion and skin tone
occasion_skin_tone_mapping = {

    "Wedding": {
    "Fair/Light": {"lipstick": "#FFB6C1", "lip_gloss": True},  # Soft pink
    "Medium/Olive": {"lipstick": "#FF69B4", "lip_gloss": True},  # Rosy pink
    "Tan": {"lipstick": "#FF1493", "lip_gloss": True},  # Bright pink
    "Dark/Deep": {"lipstick": "#8B0000", "lip_gloss": True},  # Deep red
},
"Birthday Party": {
    "Fair/Light": {"lipstick": "#FF0000", "lip_gloss": True},  # Bold red
    "Medium/Olive": {"lipstick": "#FF6347", "lip_gloss": True},  # Warm reds
    "Tan": {"lipstick": "#FF1493", "lip_gloss": True},  # Bright pink
    "Dark/Deep": {"lipstick": "#8B008B", "lip_gloss": True},  # Dark berry
},
"Dinner Date": {
    "Fair/Light": {"lipstick": "#F08080", "lip_gloss": False},  # Classic red
    "Medium/Olive": {"lipstick": "#D2691E", "lip_gloss": False},  # Rose
    "Tan": {"lipstick": "#8B0000", "lip_gloss": False},  # Rich red
    "Dark/Deep": {"lipstick": "#800000", "lip_gloss": False},  # Deep red
},
"Cocktail Party": {
    "Fair/Light": {"lipstick": "#FF1493", "lip_gloss": False},  # Bold red
    "Medium/Olive": {"lipstick": "#FF00FF", "lip_gloss": False},  # Fuchsia
    "Tan": {"lipstick": "#8B0000", "lip_gloss": False},  # Deep red
    "Dark/Deep": {"lipstick": "#800080", "lip_gloss": False},  # Dark plum
},
"Job Interview": {
    "Fair/Light": {"lipstick": "#F4A300", "lip_gloss": False},  # Nude pink
    "Medium/Olive": {"lipstick": "#D8BFD8", "lip_gloss": False},  # Soft mauve
    "Tan": {"lipstick": "#DB7093", "lip_gloss": False},  # Neutral pink
    "Dark/Deep": {"lipstick": "#8B3A3A", "lip_gloss": False},  # Warm red
},
"New Year's Eve": {
    "Fair/Light": {"lipstick": "#FF6347", "lip_gloss": True},  # Bold red
    "Medium/Olive": {"lipstick": "#8B0000", "lip_gloss": True},  # Deep berry
    "Tan": {"lipstick": "#FF4500", "lip_gloss": True},  # Classic red
    "Dark/Deep": {"lipstick": "#800000", "lip_gloss": True},  # Dark plum
},
"Valentine's Day": {
    "Fair/Light": {"lipstick": "#FF4500", "lip_gloss": True},  # Hot bright red
    "Medium/Olive": {"lipstick": "#FF4500", "lip_gloss": True},  # Hot bright red
    "Tan": {"lipstick": "#FF4500", "lip_gloss": True},  # Hot bright red
    "Dark/Deep": {"lipstick": "#FF4500", "lip_gloss": True},  # Hot bright red
},
"Christmas Party": {
    "Fair/Light": {"lipstick": "#FF6347", "lip_gloss": True},  # Classic red
    "Medium/Olive": {"lipstick": "#8B0000", "lip_gloss": True},  # Deep red
    "Tan": {"lipstick": "#FF1493", "lip_gloss": True},  # Bold red
    "Dark/Deep": {"lipstick": "#800080", "lip_gloss": True},  # Dark plum
},
"Office Party": {
    "Fair/Light": {"lipstick": "#F4A300", "lip_gloss": False},  # Nude pink
    "Medium/Olive": {"lipstick": "#F5DEB3", "lip_gloss": False},  # Soft mauve
    "Tan": {"lipstick": "#D8BFD8", "lip_gloss": False},  # Warm pink
    "Dark/Deep": {"lipstick": "#8B0000", "lip_gloss": False},  # Muted berry
},
"Girls' Night Out": {
    "Fair/Light": {"lipstick": "#FF0000", "lip_gloss": True},  # Bold red
    "Medium/Olive": {"lipstick": "#8B008B", "lip_gloss": True},  # Bold red
    "Tan": {"lipstick": "#FF1493", "lip_gloss": True},  # Bright coral
    "Dark/Deep": {"lipstick": "#800080", "lip_gloss": True},  # Dark berry
}

}

# Streamlit setup
st.title("Real-Time Lip Gloss Effect with Facial Features")

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

# Step 3: Define function to get the lipstick color and gloss effect based on skin tone and occasion
def get_lipstick_and_gloss(occasion, skin_tone):
    mapping = occasion_skin_tone_mapping.get(occasion, {})
    return mapping.get(skin_tone, {"lipstick": "#FFB6C1", "lip_gloss": False})  # Default to soft pink

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

    # Get lipstick color and gloss effect based on selected occasion and detected skin tone
    lipstick_info = get_lipstick_and_gloss(occasion, skin_tone)
    lipstick_color = lipstick_info["lipstick"]
    lip_gloss = lipstick_info["lip_gloss"]

    # Convert hex color to BGR
    lipstick_color_bgr = hex_to_bgr(lipstick_color)

    # Only focus on lips
    face_elements = ["LIP_UPPER", "LIP_LOWER"]
    colors_map = {
        "LIP_UPPER": lipstick_color_bgr,  # User-selected color for upper lip
        "LIP_LOWER": lipstick_color_bgr,  # User-selected color for lower lip
    }

    # Face connections based on lip elements
    face_connections = [face_points[idx] for idx in face_elements]
    colors = [colors_map[idx] for idx in face_elements]

    # Create an empty mask (same size as the image)
    mask = np.zeros_like(image)

    # Create a mask for facial features with color (only lips)
    mask = add_mask(mask, idx_to_coordinates=face_landmarks, face_connections=face_connections, colors=colors)

    # Apply lip gloss effect if selected
    if lip_gloss:
        gloss_color = [255, 105, 180]  # Glossy pink (BGR)
        mask = add_mask(mask, idx_to_coordinates=face_landmarks, face_connections=face_connections, colors=[gloss_color] * len(face_connections))

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
