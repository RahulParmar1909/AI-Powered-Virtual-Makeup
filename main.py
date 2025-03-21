import streamlit as st
import subprocess  # To run external Python scripts

# Title of the web app
st.title('Select a Script to Run')

# Dropdown for user selection
selection = st.selectbox("Choose an option:", ['A', 'B'])

# Run corresponding script based on the selection
if selection == 'A':
    st.write("You selected option A. Running ai.py...")
    # Run ai.py (replace with the actual path if necessary)
    result = subprocess.run(['python', 'ai.py'], capture_output=True, text=True)
    st.write(result.stdout)  # Display the output of ai.py
elif selection == 'B':
    st.write("You selected option B. Running check.py...")
    # Run check.py (replace with the actual path if necessary)
    result = subprocess.run(['python', 'check.py'], capture_output=True, text=True)
    st.write(result.stdout)  # Display the output of check.py
