# src/train_lstm.py

from .preprocess import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

# Configuración
DATA_PATHS = [
    "data/full_dataset/goemotions_1.csv",
    "data/full_dataset/goemotions_2.csv",
    "data/full_dataset/goemotions_3.csv",
]
SELECTED_EMOTIONS = ['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise', 'love']
VOCAB_SIZE = 20000
MAX_LEN = 50
EMBEDDING_DIM = 128

# Cargar y preprocesar datos
X, y = load_dataset(DATA_PATHS, SELECTED_EMOTIONS, VOCAB_SIZE, MAX_LEN)
num_classes = len(set(y))

# Definir el modelo LSTM
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Entrenar el modelo
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X, y, epochs=10, validation_split=0.2, batch_size=32, callbacks=[early_stop])

# Guardar el modelo
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")