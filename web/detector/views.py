from django.shortcuts import render
import joblib
import os

# Create your views here.

# Ruta a modelos entrenados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
print(model_path)
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Carga modelo y vectorizador
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def home(request):
    emotion = ''
    user_text = ''
    
    if request.method == 'POST':
        user_text = request.POST.get('text')
        vector = vectorizer.transform([user_text])
        emotion = model.predict(vector)[0]

    return render(request, 'detector/index.html', {
        'emotion': emotion,
        'text': user_text
    })