import cv2
from keras._tf_keras.keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_model = load_model('saved_models/best_emotion_recognition_model.keras')

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    
    face = cv2.resize(face, (48, 48))

    #save for test
    cv2.imwrite(f"screenshot_{'test'}.jpg", face)
    
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = face_gray / 255.0
    
    face_gray = np.expand_dims(face_gray, axis=-1)
    face_gray = np.expand_dims(face_gray, axis=0)
    
    return face_gray

def classify_emotion(img):
    preprocessed_img = preprocess_image(img)
    if preprocessed_img is None:
        return None, None
    predictions = emotion_model.predict(preprocessed_img)
    emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    probabilities = dict(zip(emotion_labels, predictions[0]))
    return predicted_emotion, probabilities