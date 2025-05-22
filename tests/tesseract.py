import cv2
import pytesseract

# Charger l'image
image_path = "assets/pepsi.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")
    exit()

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un seuillage pour améliorer la lisibilité
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Utiliser Tesseract pour lire le texte
text = pytesseract.image_to_string(thresh, config="--psm 6")
print("Texte détecté :", text)

# Sauvegarder l'image prétraitée (optionnel)
cv2.imwrite("output_text.jpg", thresh)