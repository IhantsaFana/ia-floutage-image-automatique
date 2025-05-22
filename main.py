import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Erreur : Impossible de charger l'image à {image_path}")
    return image

# Paramètres
IMAGE_PATH = "assets/image3.jpg"
YOLO_CONFIDENCE = 0.5
TESSERACT_CONFIG = (
    r"--oem 3 --psm 6 --dpi 300 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
LOGO_ZONE_Y = (0.4, 0.7)
LOGO_ZONE_X = (0.2, 0.8)
DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

# Chargement Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("La clé API Gemini est manquante. Vérifie ton fichier .env.")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

<<<<<<< HEAD
try:
    model = YOLO("yolo11n.pt")
    image = load_image(IMAGE_PATH)
except Exception as e:
    print(f"Erreur lors de l'initialisation : {e}")
    exit(1)
=======
# 1. Initialisation
model = YOLO("yolo11n.pt")
image_path = "assets/fanta.jpg"
image = cv2.imread(image_path)
>>>>>>> refs/remotes/origin/main

try:
    results = model.predict(source=IMAGE_PATH, classes=[0], conf=YOLO_CONFIDENCE)
    if not results or not results[0].boxes:
        raise ValueError("Aucune personne détectée dans l'image.")
except Exception as e:
    print(f"Erreur lors de la détection YOLO : {e}")
    exit(1)

# On travaille toujours sur une copie de l'image originale pour le floutage
image_to_blur = image.copy()

for i, box in enumerate(results[0].boxes.xyxy):
    try:
<<<<<<< HEAD
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2:
            print(f"Détection #{i+1} : Coordonnées invalides.")
            continue

        cropped = image[y1:y2, x1:x2]
        h, w = cropped.shape[:2]
        if h == 0 or w == 0:
            print(f"Détection #{i+1} : Région recadrée vide.")
            continue

        logo_y1 = int(h * LOGO_ZONE_Y[0])
        logo_y2 = int(h * LOGO_ZONE_Y[1])
        logo_x1 = int(w * LOGO_ZONE_X[0])
        logo_x2 = int(w * LOGO_ZONE_X[1])

        if logo_y1 >= logo_y2 or logo_x1 >= logo_x2:
            print(f"Détection #{i+1} : Zone du logo invalide.")
            continue

        logo_zone = cropped[logo_y1:logo_y2, logo_x1:logo_x2]
        if logo_zone.size == 0:
            print(f"Détection #{i+1} : Zone du logo vide après recadrage.")
            continue

        # Prétraitement OCR avancé
        gray = cv2.cvtColor(logo_zone, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        thresh = cv2.convertScaleAbs(thresh, alpha=1.2, beta=5)
        logo_zone_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # OCR avec Tesseract
        marque = pytesseract.image_to_string(
            logo_zone_resized, lang="eng", config=TESSERACT_CONFIG
        ).strip().upper()
        marque = re.sub(r'[^A-Z0-9]', '', marque)
        if not marque:
            print(f"Détection #{i+1} : Aucune marque détectée.")
            # Sauvegarde pour debug (toujours version non floutée)
            cv2.imwrite(str(DEBUG_DIR / f"logo_zone_{i}.jpg"), logo_zone)
            cv2.imwrite(str(DEBUG_DIR / f"logo_zone_processed_{i}.jpg"), thresh)
            continue

        print(f"Détection #{i+1} : Marque détectée : {marque}")

        # Sauvegarder les images intermédiaires AVANT floutage
        cv2.imwrite(str(DEBUG_DIR / f"cropped_person_{i}.jpg"), cropped)
        cv2.imwrite(str(DEBUG_DIR / f"logo_zone_{i}.jpg"), logo_zone)
        cv2.imwrite(str(DEBUG_DIR / f"logo_zone_processed_{i}.jpg"), thresh)
        cv2.imwrite(str(DEBUG_DIR / f"logo_zone_resized_{i}.jpg"), logo_zone_resized)

        # Requête Gemini (après toutes les sauvegardes)
        try:
            question = f"Est-ce que la boisson {marque} est sucrée ? Réponds uniquement par 'oui' ou 'non'."
            response = model_gemini.generate_content(question)
            reponse_texte = response.text.strip().lower()
            print(f"Réponse de Gemini : {reponse_texte}")
        except Exception as e:
            print(f"Erreur lors de la requête Gemini : {e}")
            continue

        # Floutage conditionnel (sur copie de l'image originale)
        if "oui" in reponse_texte:
            logo_x1_img = x1 + logo_x1
            logo_x2_img = x1 + logo_x2
            logo_y1_img = y1 + logo_y1
            logo_y2_img = y1 + logo_y2

            roi = image_to_blur[logo_y1_img:logo_y2_img, logo_x1_img:logo_x2_img]
            roi = cv2.GaussianBlur(roi, (51, 51), 0)
            image_to_blur[logo_y1_img:logo_y2_img, logo_x1_img:logo_x2_img] = roi

            out_path = DEBUG_DIR / f"image_floutee_{i+1}.jpg"
            cv2.imwrite(str(out_path), image_to_blur)
            print(f"Image floutée et sauvegardée sous '{out_path}'")
        else:
            out_path = DEBUG_DIR / f"image_originale_{i+1}.jpg"
            cv2.imwrite(str(out_path), image)
            print(f"Image originale sauvegardée sous '{out_path}'")
=======
        question = (
            f"Voici un texte extrait par OCR depuis un logo sur une bouteille : \"{marque}\".\n"
            f"Ignore les caractères spéciaux, les erreurs ou le bruit dans ce texte. "
            f"Tente d'identifier s'il contient ou ressemble à une marque de boisson connue.\n"
            f"Si tu reconnais une marque ou que tu peux la deviner, appelle-la : `marque_trouvée`.\n"
            f"La boisson `marque_trouvée` est-elle sucrée ?\n"
            f"Réponds uniquement par `oui` ou `non` sans aucune explication."
        )
>>>>>>> refs/remotes/origin/main

    except Exception as e:
        print(f"Erreur lors du traitement de la détection #{i+1} : {e}")
        continue

<<<<<<< HEAD
print("Traitement terminé.")
=======
    # 7. Floutage conditionnel
    if "oui" in reponse_texte:
        logo_x1 = x1 + int(w*0.15)
        logo_y1 = y1 + int(h*0.35)
        logo_x2 = x1 + int(w*0.85)
        logo_y2 = y1 + int(h*0.65)

        roi = image[logo_y1:logo_y2, logo_x1:logo_x2]
        roi = cv2.GaussianBlur(roi, (51, 51), 0)
        image[logo_y1:logo_y2, logo_x1:logo_x2] = roi

        cv2.imwrite(f"image_floutee_{i+1}.jpg", image)
        print(f"Image floutée et sauvegardée sous 'image_floutee_{i+1}.jpg'")
    else:
        cv2.imwrite("image_originale.jpg", image)
        print("Image originale sauvegardée sous 'image_originale.jpg'")
>>>>>>> refs/remotes/origin/main
