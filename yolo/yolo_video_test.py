from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("../yolo11n.pt")

# Charger la vidéo
video_path = "../assets/video1.mp4"  # Remplacer par le chemin de votre vidéo
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo est bien ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo")
    exit()

# Obtenir les propriétés de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Créer une vidéo de sortie
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Boucle sur chaque frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Effectuer la détection
    results = model.predict(source=frame, classes=0, conf=0.5)

    # Parcourir les détections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Personne", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Écrire la frame dans la vidéo de sortie
    out.write(frame)

    # Optionnel : Afficher en temps réel (commenter si non désiré)
    cv2.imshow("Détection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Appuyez sur 'q' pour quitter
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Vidéo traitée sauvegardée dans output_video.mp4")