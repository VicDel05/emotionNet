import pandas as pd

def load_and_preprocess(selected_emotions):
    # Cargar los tres CSVs
    df1 = pd.read_csv("data/full_dataset/goemotions_1.csv")
    df2 = pd.read_csv("data/full_dataset/goemotions_2.csv")
    df3 = pd.read_csv("data/full_dataset/goemotions_3.csv")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    # Filtrar solo las columnas necesarias
    emotion_columns = [col for col in df.columns if col in selected_emotions]
    df = df[['text'] + emotion_columns]

    # Eliminar filas sin ninguna emoción seleccionada
    df = df[df[emotion_columns].sum(axis=1) > 0]

    # Crear columna 'label' con la primera emoción marcada
    def get_primary_emotion(row):
        for emotion in emotion_columns:
            if row[emotion] == 1:
                return emotion
        return None  # fallback

    df['label'] = df.apply(get_primary_emotion, axis=1)

    return df['text'], df['label']
