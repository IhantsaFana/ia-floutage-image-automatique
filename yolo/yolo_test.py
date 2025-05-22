from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle
model = YOLO("../yolo11n.pt")

# Charger l'image
image_path = "../assets/image2.jpg"  # Chemin correct de votre image
image = cv2.imread(image_path)

if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")
    exit()

# Effectuer la détection
results = model.predict(source=image_path, classes=0, conf=0.5)

# Parcourir les détections
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Personne", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Sauvegarder l'image au lieu d'utiliser cv2.imshow
cv2.imwrite("output.jpg", image)
print("Résultat sauvegardé dans output.jpg")