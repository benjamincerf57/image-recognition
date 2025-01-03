from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import numpy as np
import tempfile
import os

# Preload the model and the image processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


def process_image(image):
    """
    Process an image and return the detected objects.

    Args:
        image (PIL.Image or str): Input image or path to the image.
    
    Returns:
        PIL.Image: Annotated image with detected objects.
    """

    image = Image.open(image)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load a font for the text
    try:
        font = ImageFont.truetype("arial.ttf", size=50)
    except IOError:
        font = ImageFont.load_default()

    # Go through each detection and draw boxes and text
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Convert the box to integers
        box = [round(i, 2) for i in box.tolist()]
        x_min, y_min, x_max, y_max = map(int, box)

        # Draw tge bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Add the label and score
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        
        # Calculate the width and height of the text
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Width of the text
        text_height = text_bbox[3] - text_bbox[1]  # Height of the text

        # Draw a background rectangle for the text
        text_bg = [x_min, y_min - text_height, x_min + text_width, y_min]
        
        draw.rectangle(text_bg, fill="red")

        # Write the text on the image
        draw.text((x_min, y_min - text_height), label_text, fill="white", font=font)

    return image

def process_video(video, progress_bar):
    """
    Process a video and return the detected objects as a saved mp4 file.

    Args:
        video (file): Input video file.
        progress_bar (streamlit.ProgressBar): Progress bar to track the processing progress.
    
    Returns:
        str: Path to the annotated video
    """

    # Save the video to a temporary file 
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(video.read())
    temp_file.close()

    # Load the video 
    cap = cv2.VideoCapture(temp_file.name)

    # Check if the video is opened
    if not cap.isOpened():
        raise ValueError("Could not open video")

    # Take the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count_total / fps

    # Ensure the video is not too long (less than 15 seconds)
    if duration > 15:
        raise ValueError("The video is too long. Please upload a video of less than 15 seconds.")

    # Create a directory to save the video
    output_dir = 'results/videos'
    os.makedirs(output_dir, exist_ok=True)
    
    # output video path
    output_video_path = os.path.join(output_dir, 'output_video.mp4')

    # Create a video writer
    target_fps = 10 # FPS for the output video
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, 
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0  # Counter for the frames
    frame_interval = int(fps / target_fps)  # Frame interval to reduce the FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of the video

        # Only process every `frame_interval` frames
        if frame_count % frame_interval == 0:
            # Convert the frame to an image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Pass the image to the model
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            # Post-process the results
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            # Load a font for the text
            try:
                font = ImageFont.truetype("arial.ttf", size=15)
            except IOError:
                font = ImageFont.load_default()

            # Create a drawing object
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

            # Add the annotated frame to the video
            annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            out.write(annotated_frame)

        frame_count += 1
        if frame_count % 10 == 0:
            progress = (frame_count / frame_count_total) * 100  # frame_count_total being the total number of frames in the video
            print(f"{round(progress, 2)}% of the video is processed...")

        # Update the progress bar
        progress = frame_count / frame_count_total
        progress_bar.progress(progress)

    cap.release()
    out.release()
    print("All the frames are annotated and saved.")

    # Retourner le chemin du fichier vidéo sauvegardé
    return output_video_path    

def process_webcam(fps=1):
    """
    Process the webcam feed and return the detected objects.

    Args:
        fps (int): Frames per second to process the webcam feed.

    Yields:
        np.array: Annotated frame with detected objects.
    """

    # Initalize the webcam
    cap = cv2.VideoCapture(0)
    
    # Set the webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load a font for the text
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()
    
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to an image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Pass the image to the model
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process the results
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Annotate the image with the detected objects
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

        # Convert the image back to the BGR format
        annotated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Wait to display the frame
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        # Yield the annotated frame
        yield annotated_frame

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()