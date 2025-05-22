from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("yolo11n.pt")

# Charger l'image
image_path = "assets/pepsi.jpg"  # Remplacer par votre image ou "0" pour webcam
image = cv2.imread(image_path) if isinstance(image_path, str) else cv2.VideoCapture(image_path)

# Effectuer la détection
results = model.predict(source=image_path, classes=0, conf=0.5)

# Parcourir les détections
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Personne", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Afficher le résultat
cv2.imshow("Détection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()