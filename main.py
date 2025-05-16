from src.predict import predecir

while True:
    text = input("Enter a sentence in English (or 'exit'): ")
    if text.lower() == "exit":
        break
    emotion = predecir(text)
    print(f"Predicted emotion: {emotion}\n")

# texto = input("Introduce un texto para detectar la emoción: ")
# emocion = predecir(texto)
# print(f"Emoción detectada: {emocion}")
