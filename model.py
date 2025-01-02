from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import numpy as np
import tempfile
import os
import subprocess

# Preload the model and the image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def process_image(image):
    '''
    Process an image and return the detected objects.
    '''

    image = Image.open(image)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Créer un objet de dessin pour l'image
    draw = ImageDraw.Draw(image)

    # Optionnel : Définir une police pour le texte (si disponible)
    try:
        font = ImageFont.truetype("arial.ttf", size=50)
    except IOError:
        font = ImageFont.load_default()

    # Parcourir les détections et dessiner les boîtes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Convertir les coordonnées des boîtes en entiers
        box = [round(i, 2) for i in box.tolist()]
        x_min, y_min, x_max, y_max = map(int, box)

        # Dessiner le rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Ajouter le label et le score
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        
        # Calculer les dimensions du texte (bbox)
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Largeur du texte
        text_height = text_bbox[3] - text_bbox[1]  # Hauteur du texte

        # Dessiner le fond du texte (rectangle rouge derrière le texte)
        text_bg = [x_min, y_min - text_height, x_min + text_width, y_min]
        
        draw.rectangle(text_bg, fill="red")

        # Dessiner le texte
        draw.text((x_min, y_min - text_height), label_text, fill="white", font=font)

    return image

def process_video(video):
    '''
    Process a video and return the detected objects as a saved mp4 file.
    '''
    # Sauvegarder la vidéo téléchargée dans un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(video.read())
    temp_file.close()

    # Charger la vidéo à partir du fichier temporaire
    cap = cv2.VideoCapture(temp_file.name)

    # Check if the video is opened
    if not cap.isOpened():
        raise ValueError("Could not open video")

    # FPS original de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count_total / fps

    # N'executer le modèle que sur les vidéos de moins de 15 secondes
    if duration > 15:
        raise ValueError("The video is too long. Please upload a video of less than 15 seconds.")

    # Créer le dossier de sortie si il n'existe pas
    output_dir = 'results/videos'
    os.makedirs(output_dir, exist_ok=True)
    
    # Chemin du fichier de sortie
    output_video_path = os.path.join(output_dir, 'output_video.mp4')

    # Créer un flux vidéo pour la vidéo annotée
    target_fps = 10 # FPS cible pour la vidéo annotée
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, 
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0  # Compteur pour suivre le nombre de frames traitées
    frame_interval = int(fps / target_fps)  # Interval de frames pour limiter le FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        # Traiter uniquement les frames nécessaires
        if frame_count % frame_interval == 0:
            # Convertir la frame OpenCV en PIL.Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Passer l'image au modèle
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Post-traitement des résultats
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Optionnel : Charger une police pour le texte
            try:
                font = ImageFont.truetype("arial.ttf", size=15)
            except IOError:
                font = ImageFont.load_default()

            # Annoter l'image
            draw = ImageDraw.Draw(image)

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                x_min, y_min, x_max, y_max = map(int, box)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_bg = [x_min, y_min - text_height, x_min + text_width, y_min]
                draw.rectangle(text_bg, fill="red")
                draw.text((x_min, y_min - text_height), label_text, fill="white", font=font)

            # Ajouter la frame annotée à la vidéo
            annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out.write(annotated_frame)

        frame_count += 1
        if frame_count % 10 == 0:
            progress = (frame_count / frame_count_total) * 100  # frame_count_total étant le nombre total de frames
            print(f"{round(progress, 2)}% de la vidéo traitée...")

    cap.release()
    out.release()
    print("Toutes les frames ont été annotées et enregistrées.")

    # Retourner le chemin du fichier vidéo sauvegardé
    return output_video_path    

def process_webcam():
    # Simule une prédiction
    pass

