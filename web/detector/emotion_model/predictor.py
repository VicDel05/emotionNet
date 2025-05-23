import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Ruta segura al archivo
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Lista de etiquetas (ajústala según tu dataset)
emotion_labels = ['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise', 'love']

MAX_LEN = 100  # el mismo usado en entrenamiento

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # quita signos
    return text.strip()

def predict_emotion(text):
    texto = clean_text(text)
    seq = tokenizer.texts_to_sequences([texto])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0]
    label_idx = np.argmax(pred)
    return emotion_labels[label_idx], float(pred[label_idx])
