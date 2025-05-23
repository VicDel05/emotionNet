# src/train_lstm.py

from .preprocess import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import pickle

# Configuración
DATA_PATHS = [
    "data/full_dataset/goemotions_1.csv",
    "data/full_dataset/goemotions_2.csv",
    "data/full_dataset/goemotions_3.csv",
]
X_train, X_val, y_train, y_val, tokenizer, num_classes = load_dataset(
    DATA_PATHS,
    selected_emotions=['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise', 'love'],  # o usa `None` para todas
    num_words=10000,
    max_len=100
)

SELECTED_EMOTIONS = ['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise', 'love']
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128

# Definir el modelo LSTM
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32, return_sequences=False), # Cantidad de neuronas
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Entrenar el modelo
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, validation_split=0.1, batch_size=32, callbacks=[early_stop])

# Guardar el modelo
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)