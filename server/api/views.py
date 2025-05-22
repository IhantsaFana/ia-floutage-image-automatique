from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
from django.conf import settings
from django.core.files.storage import default_storage

# Charger Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# Charger YOLO une fois
model = YOLO("yolo11n.pt")

class DetectionAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({"error": "Image manquante"}, status=status.HTTP_400_BAD_REQUEST)

        # Sauvegarder temporairement l'image
        temp_path = default_storage.save('tmp/' + file_obj.name, file_obj)
        image_path = os.path.join(settings.MEDIA_ROOT, temp_path)

        image = cv2.imread(image_path)
        if image is None:
            return Response({"error": "Impossible de lire l'image"}, status=status.HTTP_400_BAD_REQUEST)

        # Paramètres (à ajuster)
        YOLO_CONFIDENCE = 0.5
        TESSERACT_CONFIG = (
            r"--oem 3 --psm 6 --dpi 300 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        LOGO_ZONE_Y = (0.4, 0.7)
        LOGO_ZONE_X = (0.2, 0.8)

        results = model.predict(source=image_path, classes=[0], conf=YOLO_CONFIDENCE)
        if not results or not results[0].boxes:
            return Response({"error": "Aucune personne détectée"}, status=status.HTTP_400_BAD_REQUEST)

        image_to_blur = image.copy()
        floutage_effectue = False

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h == 0 or w == 0:
                continue

            logo_y1 = int(h * LOGO_ZONE_Y[0])
            logo_y2 = int(h * LOGO_ZONE_Y[1])
            logo_x1 = int(w * LOGO_ZONE_X[0])
            logo_x2 = int(w * LOGO_ZONE_X[1])

            logo_zone = cropped[logo_y1:logo_y2, logo_x1:logo_x2]
            gray = cv2.cvtColor(logo_zone, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
            )
            thresh = cv2.convertScaleAbs(thresh, alpha=1.2, beta=5)
            logo_zone_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            marque = pytesseract.image_to_string(
                logo_zone_resized, lang="eng", config=TESSERACT_CONFIG
            ).strip().upper()
            marque = re.sub(r'[^A-Z0-9]', '', marque)
            
            print("Marque détectée :", marque)
            
            if not marque:
                continue

            # Interroger Gemini
            question = f"Est-ce que la boisson {marque} est sucrée ? Réponds uniquement par 'oui' ou 'non'."
            try:
                response = model_gemini.generate_content(question)
                reponse_texte = response.text.strip().lower()
            except Exception:
                continue

            if "oui" in reponse_texte:
                logo_x1_img = x1 + logo_x1
                logo_x2_img = x1 + logo_x2
                logo_y1_img = y1 + logo_y1
                logo_y2_img = y1 + logo_y2

                roi = image_to_blur[logo_y1_img:logo_y2_img, logo_x1_img:logo_x2_img]
                roi = cv2.GaussianBlur(roi, (51, 51), 0)
                image_to_blur[logo_y1_img:logo_y2_img, logo_x1_img:logo_x2_img] = roi
                floutage_effectue = True

        # Sauvegarder l'image finale
        output_path = os.path.join(settings.MEDIA_ROOT, "output.jpg")
        cv2.imwrite(output_path, image_to_blur if floutage_effectue else image)

        # Supprimer l'image temporaire
        default_storage.delete(temp_path)

        # Retourner le résultat (URL ou base64)
        with open(output_path, "rb") as f:
            import base64
            img_base64 = base64.b64encode(f.read()).decode()

        return Response({
            "floutage": floutage_effectue,
            "image_base64": img_base64,
        })
