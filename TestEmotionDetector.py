import cv2
import sqlite3
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import model_from_json
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('face_emotion_database.db')
cursor = conn.cursor()

# Create a table to store face information
cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emotion TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# Load the trained model from .json and .h5 files
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model/emotion_model.h5')

# Load cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Tkinter GUI
root = tk.Tk()
root.title("Facial Expression Recognition")
root.configure(bg="lightpink")

# Create a label to display emotions
emotion_label = tk.Label(root, text="Detected Emotion: ")
emotion_label.pack(pady=10)

# Open video capture
video_capture = cv2.VideoCapture(0)

#Create a text file for saving database information
text_file= open('database_information.txt','w')


def detect_emotion():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI and preprocess it
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))  # Ensure the input size matches your model's expected size
        roi = np.expand_dims(roi, axis=0) / 255.0  # Normalize input

        # Use the trained model to predict facial expression
        prediction = loaded_model.predict(roi)
        emotion_label = get_emotion_label(np.argmax(prediction))

        # Draw rectangle around the face and display emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Insert face information into the database
        cursor.execute('INSERT INTO face_emotions (emotion) VALUES (?)', (emotion_label,))
        conn.commit()

        #Save information to text file
        text_file.write(f'{emotion_label} - {get_current_timestamp()}\n')

    # Display the frame with emotion
    cv2.imshow('Video', frame)
    root.after(10, detect_emotion)  # Call the function again after 10 milliseconds


def get_emotion_label(emotion_index):
    # Map model output index to emotion label (customize based on your model)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return emotion_mapping.get(emotion_index, 'Unknown')

def get_current_timestamp():
    return cursor.execute('SELECT CURRENT_TIMESTAMP').fetchone()[0]


# Button to start emotion detection
start_button = tk.Button(root, text="Start Emotion Detection", command=detect_emotion)
start_button.pack(pady=10)

# Exit button
exit_button = tk.Button(root, text="Exit", command=lambda:on_exit())
exit_button.pack(pady=10)

def on_exit():
    text_file.close()
    root.destroy()

# Tkinter main loop
root.mainloop()

# Release resources
video_capture.release()
cv2.destroyAllWindows()
conn.close()

