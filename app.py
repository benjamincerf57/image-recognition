import streamlit as st
from model import process_image, process_video, process_webcam
import time

# Titre et description de l'app
st.title("Image Recognition with DETR")
st.write("Bienvenue dans notre application de détection d'objets ! Vous pouvez tester des images, des vidéos ou la webcam.")

# Créer des onglets
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Webcam"])

# Onglet Image
with tab1:
    st.header("Test sur une image")
    uploaded_image = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg"])
    if uploaded_image is not None:
        # Processer l'image
        image = process_image(uploaded_image)
        st.image(image, caption="Image traitée", use_column_width=True)

# Onglet Vidéo
with tab2:
    st.header("Test sur une vidéo")
    uploaded_video = st.file_uploader("Téléchargez une vidéo", type=["mp4"])
    
    if uploaded_video is not None:
        start_time = time.time()

        # Créer la barre de progression
        progress_bar = st.progress(0)
        # Processer la vidéo avec l'état de progression
        video = process_video(uploaded_video, progress_bar)
        # Afficher la vidéo traitée
        st.video(video)

        elapsed_time = time.time() - start_time
        # Display the processing time
        st.write(f"Processing time: {elapsed_time:.2f} seconds")


# Onglet Webcam
with tab3:
    st.header("Test via la webcam")
    if st.button("Activer la Webcam"):
        # Processer la webcam
        process_webcam()
