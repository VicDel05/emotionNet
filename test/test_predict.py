from src.predict import predecir

def test_prediccion():
    ejemplos = {
        "Estoy muy feliz": "joy",
        "Esto es horrible": "anger",
        "Me siento triste": "sadness"
    }

    for texto, emocion_esperada in ejemplos.items():
        emocion_predicha = predecir(texto)
        print(f"Texto: {texto} | Esperada: {emocion_esperada} | Detectada: {emocion_predicha}")

if __name__ == "__main__":
    test_prediccion()
