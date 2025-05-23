# src/preprocess_lstm.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""  # puedes usar np.nan si quieres filtrar después
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # eliminar URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # eliminar puntuación y números
    text = re.sub(r"\s+", " ", text).strip()  # eliminar espacios extras
    return text

def load_dataset(csv_paths, selected_emotions, num_words=20000, max_len=100):
    # Combinar todos los CSV en uno solo
    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

    print(df)
    # Filtrar emociones si se desea
    df = df[df[selected_emotions].sum(axis=1) > 0]
    print(df)
    
    # Limpiar texto
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.strip() != ""]
    print(df)
    # Convertir one-hot a etiquetas
    df['label'] = df[selected_emotions].idxmax(axis=1)

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)

    # Etiquetas como vectores categóricos
    y = to_categorical(df['label'], num_classes=num_classes)


    # Tokenizar texto
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # División de entrenamiento/validación
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)
    print(X_train)
    print(X_val)
    print(y_train)
    print(y_val)

    # Guardar tokenizer y encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(tokenizer, "models/tokenizer.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    return X_train, X_val, y_train, y_val, tokenizer, num_classes
