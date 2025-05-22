from django.urls import path
from .views import DetectionAPIView

urlpatterns = [
    path('detect/', DetectionAPIView.as_view(), name='detect'),
]
