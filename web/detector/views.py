from django.shortcuts import render
from django.http import JsonResponse
from .emotion_model.predictor import predict_emotion
import joblib
import os

# Create your views here.


def predict_view(request):
    if request.method == "POST":
        text = request.POST.get("text", "")
        if text:
            label, confidence = predict_emotion(text)
            return JsonResponse({
                "emotion": label,
                "confidence": round(confidence * 100, 2)
            })
    return JsonResponse({"error": "Text not provided."}, status=400)

def home(request):

    return render(request, 'detector/index.html', {})