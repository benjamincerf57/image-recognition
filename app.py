import streamlit as st
from model import process_image, process_video, process_webcam

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
        # Processer la vidéo
        video = process_video(uploaded_video)
        st.video(video)

# Onglet Webcam
with tab3:
    st.header("Test via la webcam")
    if st.button("Activer la Webcam"):
        # Processer la webcam
        process_webcam()
