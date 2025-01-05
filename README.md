# Building and Deploying Web Application for Object Detection Model

## Description
This is an object detection web application built using Streamlit and a pre-trained DETR model from Facebook (references below), which builds upon architectures like ResNet. The app allows users to upload images or videos, and it detects objects in them with a confidence score. The application is deployed on Google Cloud with a Docker image, providing easy access through a web interface.

ResNet Model: The ResNet model used in this project was introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in the paper Deep Residual Learning for Image Recognition. You can consult their paper here: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). Here is the GitHub repo of the model: [DETR](https://github.com/facebookresearch/detr)

## Technologies Used

This project utilizes the following technologies:

- **Streamlit**: A Python library used to create the interactive web interface for image uploads and object detection.
- **Docker**: Containerization tool to package the application, making it easy to deploy and run consistently across different environments.
- **Google Cloud**: Cloud platform used for deploying the application, hosting the model, and scaling it to handle user traffic.
- **DETR (DEtection TRansformers)**: The object detection model used in the project, based on transformers. It is implemented via the Hugging Face library.
- **Pillow**: Python Imaging Library for handling image manipulation and text rendering on the processed images.

## Accessing the Deployed Web Application
Once the application is deployed, you can access it through the following link: [Image recognition App](https://img-recognition-app-648581765281.europe-west1.run.app/)

### How to Use
Upload an Image or Video: Click the "Upload Image" or "Upload Video" button to select an image or video from your local device.

Detect Objects: After uploading, the model will process the image/video and highlight detected objects with bounding boxes. The corresponding class names and detection confidence will be displayed near each object.

Note: Please ensure the file is in a supported format (e.g., JPG, PNG, MP4). The uploaded videos must be shorter than 15 seconds. Videos can take up to 5 minutes to be processed.

## Project Repository Overview

The repository contains the following structure:

- **Main Folder**: All files related to the deployed web application are in the main folder. This includes:
  - `app.py`: The main Streamlit app that runs the object detection model.
  - `model.py`: The script containing the model logic, including loading the pretrained DETR model and performing object detection.
  - `Dockerfile`: The file used to build the Docker image for deploying the web app.
  - `requirements.txt`: The dependencies required to run the app.

- **Webcam Version Folder (`webcam_version`)**: 
  This folder contains an alternative version of the app that supports real-time object detection via your webcam. In this version, you can interact with the model directly through a live video stream from your webcam.
  - `webcam_app.py`: The Streamlit app used to display real-time object detection.
  - **Note**: This version has not been deployed yet.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app**

    ```bash
    streamlit run app.py
    ```

    The app should open in your browser.







