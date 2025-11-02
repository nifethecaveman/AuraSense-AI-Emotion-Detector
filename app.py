# app.py
from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import sqlite3
from datetime import datetime
import base64

app = Flask(__name__)
model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def init_db():
    conn = sqlite3.connect('AuraDB.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        image BLOB,
        emotion TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

def preprocess_image(img_bytes):
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    img_bytes = None

    if 'image' in request.files:
        # Uploaded image
        file = request.files['image']
        img_bytes = file.read()
    elif 'webcam_image' in request.form:
        # Webcam image (base64)
        data_url = request.form['webcam_image']
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)

    if img_bytes:
        img = preprocess_image(img_bytes)
        prediction = model.predict(img)
        emotion = emotion_labels[np.argmax(prediction)]

        # Save to DB
        conn = sqlite3.connect('AuraDB.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, image, emotion, timestamp) VALUES (?, ?, ?, ?)",
                  (name, sqlite3.Binary(img_bytes), emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        return render_template('index.html', emotion=emotion)
    
    

    return render_template('index.html', emotion="No image received")

if __name__ == '__main__':
    app.run(debug=True)
