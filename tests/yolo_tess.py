import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# 1. Initialisation et gestion des erreurs
def load_image(image_path):
    """Charge une image et gère les erreurs."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Erreur : Impossible de charger l'image à {image_path}")
    return image

# Paramètres configurables
IMAGE_PATH = "assets/pepsi.jpg"
YOLO_CONFIDENCE = 0.5
TESSERACT_CONFIG = r"--oem 3 --psm 6 --dpi 300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # PSM 6 pour une ligne de texte
LOGO_ZONE_Y = (0.4, 0.7)  # Ajusté : 40% à 70% de la hauteur pour mieux cibler le t-shirt
LOGO_ZONE_X = (0.2, 0.8)  # Ajusté : 20% à 80% de la largeur pour centrer le texte

# Charger le modèle et l'image
try:
    model = YOLO("yolo11n.pt")
    image = load_image(IMAGE_PATH)
except Exception as e:
    print(f"Erreur lors de l'initialisation : {e}")
    exit(1)

# 2. Détection de la personne
try:
    results = model.predict(source=IMAGE_PATH, classes=[0], conf=YOLO_CONFIDENCE)
    if not results or not results[0].boxes:
        raise ValueError("Aucune personne détectée dans l'image.")
except Exception as e:
    print(f"Erreur lors de la détection YOLO : {e}")
    exit(1)

# 3. Parcourir les détections
for i, box in enumerate(results[0].boxes.xyxy):
    try:
        # Extraire les coordonnées
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2:
            print(f"Détection #{i+1} : Coordonnées invalides (x1={x1}, x2={x2}, y1={y1}, y2={y2}).")
            continue

        # Recadrer la région de la personne
        cropped = image[y1:y2, x1:x2]
        h, w = cropped.shape[:2]
        if h == 0 or w == 0:
            print(f"Détection #{i+1} : Région recadrée vide (hauteur={h}, largeur={w}).")
            continue

        # 4. Extraction de la zone du logo (t-shirt)
        logo_y1 = int(h * LOGO_ZONE_Y[0])
        logo_y2 = int(h * LOGO_ZONE_Y[1])
        logo_x1 = int(w * LOGO_ZONE_X[0])
        logo_x2 = int(w * LOGO_ZONE_X[1])

        if logo_y1 >= logo_y2 or logo_x1 >= logo_x2:
            print(f"Détection #{i+1} : Zone du logo invalide (y1={logo_y1}, y2={logo_y2}, x1={logo_x1}, x2={logo_x2}).")
            continue

        logo_zone = cropped[logo_y1:logo_y2, logo_x1:logo_x2]
        if logo_zone.size == 0:
            print(f"Détection #{i+1} : Zone du logo vide après recadrage.")
            continue

        # 5. Prétraitement OCR amélioré
        gray = cv2.cvtColor(logo_zone, cv2.COLOR_BGR2GRAY)
        
        # Ajouter un flou pour réduire le bruit
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Seuillage adaptatif avec ajustement
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Optionnel : Contraste et redimensionnement pour améliorer la détection
        thresh = cv2.convertScaleAbs(thresh, alpha=1.2, beta=5)
        logo_zone_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # 6. OCR avec Tesseract
        marque = pytesseract.image_to_string(
            logo_zone_resized, lang="eng", config=TESSERACT_CONFIG
        ).strip()

        if not marque:
            print(f"Détection #{i+1} : Aucune marque détectée.")
            # Sauvegarder pour débogage
            cv2.imwrite(f"logo_zone_{i}.jpg", logo_zone)
            cv2.imwrite(f"logo_zone_processed_{i}.jpg", thresh)
            continue

        print(f"Détection #{i+1} : Marque détectée : {marque}")

        # Sauvegarder les images intermédiaires pour vérification
        cv2.imwrite(f"cropped_person_{i}.jpg", cropped)
        cv2.imwrite(f"logo_zone_{i}.jpg", logo_zone)
        cv2.imwrite(f"logo_zone_processed_{i}.jpg", thresh)
        cv2.imwrite(f"logo_zone_resized_{i}.jpg", logo_zone_resized)

    except Exception as e:
        print(f"Erreur lors du traitement de la détection #{i+1} : {e}")
        continue

print("Traitement terminé.")