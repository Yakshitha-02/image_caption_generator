# Image-Caption-Generator

An AI-powered image description application that uses a pretrained ResNet50 deep learning model to identify objects in images and generate human-readable captions.

## Overview

This project utilizes transfer learning with the ResNet50 model trained on the ImageNet dataset. Users can upload an image through a web interface, and the system automatically predicts the primary object present in the image and generates a descriptive caption.

The application is built using FastAPI and TensorFlow, providing a simple and efficient image-to-text experience.

## Features

* Upload images through a web interface
* Automatic object recognition
* Deep learning inference using ResNet50
* Human-readable caption generation
* FastAPI-powered backend
* Lightweight and easy to deploy

## Tech Stack

* Python
* FastAPI
* TensorFlow
* Keras
* ResNet50
* NumPy
* Pillow (PIL)
* HTML

## Project Workflow

1. User uploads an image.
2. Image is resized and preprocessed.
3. ResNet50 extracts features and predicts the object class.
4. The top prediction is decoded.
5. A descriptive caption is generated.
6. Caption is returned to the user.

## Project Structure

```text
Image-Caption-Generator/
│
├── main.py
├── index.html
├── requirements.txt
├── static/
├── screenshots/
└── README.md
```

## Sample Output

### Input

Image containing a dog.

### Output

```text
A photo of a golden retriever
```

### Input

Image containing a cat.

### Output

```text
A photo of a Persian cat
```

## Installation

### Clone Repository

```bash
git clone https://github.com/Yakshitha-02/Image-Caption-Generator.git
cd Image-Caption-Generator
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
uvicorn main:app --reload
```

### Open Browser

```text
http://127.0.0.1:8000
```

## Future Improvements

* CNN-LSTM based caption generation
* Transformer-based image captioning
* BLIP image captioning model integration
* Multi-object scene descriptions
* Cloud deployment
* Mobile application support

## Learning Outcomes

* Transfer Learning
* Deep Learning Inference
* Computer Vision
* FastAPI Development
* TensorFlow & Keras
* REST API Development

## Author

**Yakshitha Senapathi**

* LinkedIn: https://linkedin.com/in/yakshitha02
* GitHub: https://github.com/Yakshitha-02
