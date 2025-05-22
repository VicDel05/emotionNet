# src/preprocess_lstm.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os

def load_dataset(csv_paths, selected_emotions, num_words=20000, max_len=50):
    # Combinar todos los CSV en uno solo
    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

    # Filtrar solo una emoción por fila
    emotion_columns = selected_emotions
    df['label'] = df[emotion_columns].idxmax(axis=1)
    df = df[df[emotion_columns].sum(axis=1) == 1]

    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()

    # Tokenización
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Guardar tokenizer y encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(tokenizer, "models/tokenizer.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    return padded, y
