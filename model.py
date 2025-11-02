# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Load CSV
data = pd.read_csv('fer2013.csv')

# Map emotion labels
emotion_map = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# Convert pixels to numpy arrays
pixels = data['pixels'].tolist()
faces = np.array([np.fromstring(p, sep=' ').reshape(48, 48) for p in pixels])
faces = np.expand_dims(faces, -1) / 255.0  # Normalize and add channel

# One-hot encode emotions
emotions = to_categorical(data['emotion'], num_classes=7)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))
model.save('emotion_model.h5')