from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle
model = YOLO("yolo11n.pt")

# Charger l'image
image_path = "assets/pepsi.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")
    exit()

# Effectuer la détection
results = model.predict(source=image_path, classes=[0, 39], conf=0.5)  # 0 = personne, 39 = t-shirt

# Parcourir les détections
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = "T-shirt" if class_id == 39 else "Personne"
        # Dessiner le rectangle et le label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Sauvegarder l'image
cv2.imwrite("output.jpg", image)
print("Résultat sauvegardé dans output.jpg")