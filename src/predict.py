# src/predict_lstm.py

import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Cargar recursos
model = load_model("models/lstm_model.keras")
tokenizer = joblib.load("models/tokenizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Configuración
MAX_LEN = 100  # Debe coincidir con el entrenamiento

def predict_emotion(text):
    # Preprocesamiento
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

    # Predicción
    prediction = model.predict(padded)
    predicted_label = np.argmax(prediction)

    emotion = label_encoder.inverse_transform([predicted_label])[0]
    confidence = np.max(prediction)

    return emotion, confidence

# Para uso desde consola
if __name__ == "__main__":
    while True:
        text = input("Escribe una frase en inglés (o 'exit' para salir): ")
        if text.lower() == "exit":
            break
        emotion, confidence = predict_emotion(text)
        print(f"Emoción detectada: {emotion} ({confidence:.2%} confianza)\n")

