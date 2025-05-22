from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle de segmentation
model = YOLO("yolo11m-seg.pt")

# Charger l'image
image_path = "assets/pepsi.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")
    exit()

# Effectuer la détection et segmentation (uniquement pour le t-shirt)
results = model.predict(source=image_path, classes=[39], conf=0.5)  # 39 = t-shirt

# Parcourir les détections
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Dessiner le rectangle et le label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "T-shirt", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Optionnel : Dessiner les masques de segmentation
    if result.masks:
        for mask in result.masks.data:
            mask = mask.cpu().numpy() * 255
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            image[mask > 0] = (0, 255, 255)  # Colorer la zone segmentée en cyan

# Sauvegarder l'image
cv2.imwrite("output_tshirt_seg.jpg", image)
print("Résultat sauvegardé dans output_tshirt_seg.jpg")