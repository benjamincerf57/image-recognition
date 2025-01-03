import streamlit as st
from model import process_image, process_video, process_webcam
import time
import cv2

# Title and description of the app
st.title("Image Recognition with DETR")
st.write("Welcome in our object detection app ! You can try to detect objects and persons on your images, videos or directly using your webcam !")

# Create 3 different tabs
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Webcam"])

# Image tab
with tab1:
    st.header("Test with an image 📷")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg"])
    if uploaded_image is not None:
        # Image processing
        image = process_image(uploaded_image)
        st.image(image, caption="Processed image", use_column_width=True)

# Video tab
with tab2:
    st.header("Test with a video 🎥")
    st.write("The app will process the video and display the detected objects, reducing the quality and the FPS to 10 for a not too long processing time.")
    uploaded_video = st.file_uploader("Upload a video (less than 15 seconds)", type=["mp4"])
    
    if uploaded_video is not None:
        start_time = time.time()

        # Create a progress bar
        progress_bar = st.progress(0)
        # Process the video
        video = process_video(uploaded_video, progress_bar)
        # Display the processed video
        st.video(video)

        elapsed_time = time.time() - start_time
        # Display the processing time
        st.write(f"Processing time: {elapsed_time:.2f} seconds")


# Webcam tab
with tab3:
    st.header("Live object detection using your webcam 📹")
    
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")
    
    # Start the webcam if the start button is clicked
    if start_button:
        stframe = st.empty()  # Placeholder to display the webcam frames
        
        for annotated_frame in process_webcam(fps=10):
            # Convert the annotated frame to the BGR format
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # Stop the webcam if the stop button is clicked
            if stop_button:
                break
