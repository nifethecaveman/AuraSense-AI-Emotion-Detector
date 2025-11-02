# AI Emotion Detector

A web-based application that detects human emotions from facial images using deep learning. Built with Flask for the backend and TensorFlow/Keras for emotion classification, the app allows users to upload an image or capture one using their webcam. It then predicts the emotion and stores the result in a local SQLite database.

## 🔍 Features

- 🎯 Emotion recognition: Angry, Happy, Sad, Fear, Disgust, Surprise, Neutral
- 📸 Webcam integration for live image capture
- 🖼️ Image upload support
- 🧠 CNN model built with TensorFlow/Keras
- 🗃️ SQLite database logging
- 🌐 Flask-powered web interface

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **ML Framework**: TensorFlow, Keras, NumPy, OpenCV
- **Database**: SQLite

## 🚀 How It Works

1. User accesses the web interface.
2. Uploads or captures an image.
3. Flask backend receives and preprocesses the image.
4. The image is passed to a trained CNN model.
5. The predicted emotion is displayed and logged.
