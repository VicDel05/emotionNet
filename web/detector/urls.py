from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path("predict/", views.predict_view, name="predict"),
]