from src.predict import predecir

while True:
    text = input("Enter a sentence in English (or 'exit'): ")
    if text.lower() == "exit":
        break
    emotion = predecir(text)
    print(f"Predicted emotion: {emotion}\n")
