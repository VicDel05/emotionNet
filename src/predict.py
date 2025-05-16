import sys
import joblib

def predecir(texto):
    model = joblib.load('models/emotion_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    vec = vectorizer.transform([texto])
    return model.predict(vec)[0]

if __name__ == "__main__":
    texto = " ".join(sys.argv[1:])  # permite recibir texto por línea de comandos
    print(f"Texto: {texto}")
    print(f"Emoción detectada: {predecir(texto)}")
