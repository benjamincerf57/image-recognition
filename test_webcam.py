import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np


if __name__ == "__main__":
    # Le reste de ton code ici

    # Charger le modèle et le processeur
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # Initialiser la webcam
    cap = cv2.VideoCapture(0)

    # Définir la résolution de la webcam pour réduire la charge
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largeur de la vidéo (par exemple 640px)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Hauteur de la vidéo (par exemple 480px)

    # Optionnel : Définir une police pour le texte (si disponible)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    fps = 10  # Limiter à 10 FPS

    while True:
        # Lire une frame de la webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la frame OpenCV en image PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Passer l'image au modèle
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-traitement des résultats
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

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

        # Convertir l'image annotée PIL en format OpenCV pour affichage
        annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Afficher la frame annotée
        cv2.imshow("Webcam - Detr Object Detection", annotated_frame)

        # Limiter à 10 FPS (attendre 100 ms)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Libérer la caméra et fermer les fenêtres
    cap.release()
    cv2.destroyAllWindows()
