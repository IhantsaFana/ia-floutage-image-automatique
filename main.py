from ultralytics import YOLO
import cv2
import pytesseract
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configuration Gemini
if not api_key:
    raise ValueError("La clé API Gemini est manquante. Vérifie ton fichier .env.")
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# 1. Initialisation
model = YOLO("yolo11n.pt")
image_path = "assets/fanta.jpg"
image = cv2.imread(image_path)

# 2. Détection de la personne
results = model.predict(source=image_path, classes=0, conf=0.5)

for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    h, w = cropped.shape[:2]

    # 3. Extraction de la zone du logo
    logo_zone = cropped[int(h*0.35):int(h*0.65), int(w*0.15):int(w*0.85)]

    # 4. Prétraitement OCR
    gray = cv2.cvtColor(logo_zone, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15
    )

    # 5. OCR
    custom_config = r'--oem 3 --psm 7'
    marque = pytesseract.image_to_string(thresh, lang='eng', config=custom_config).strip()
    print(f"Marque détectée : {marque}")

    if not marque:
        print("Aucune marque détectée, on passe à la suivante.")
        continue

    # 6. Requête Gemini
    try:
        question = (
            f"Voici un texte extrait par OCR depuis un logo sur une bouteille : \"{marque}\".\n"
            f"Ignore les caractères spéciaux, les erreurs ou le bruit dans ce texte. "
            f"Tente d'identifier s'il contient ou ressemble à une marque de boisson connue.\n"
            f"Si tu reconnais une marque ou que tu peux la deviner, appelle-la : `marque_trouvée`.\n"
            f"La boisson `marque_trouvée` est-elle sucrée ?\n"
            f"Réponds uniquement par `oui` ou `non` sans aucune explication."
        )

        response = model_gemini.generate_content(question)
        reponse_texte = response.text.strip().lower()
        print(f"Réponse de Gemini : {reponse_texte}")
    except Exception as e:
        print(f"Erreur lors de la requête Gemini : {e}")
        continue

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
