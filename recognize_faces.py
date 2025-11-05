import cv2
import numpy as np
import os

# Folder for saved faces
KNOWN_DIR = "known_faces"
os.makedirs(KNOWN_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_recognizer():
    faces, labels = [], []
    label_names = []
    for i, file in enumerate(os.listdir(KNOWN_DIR)):
        path = os.path.join(KNOWN_DIR, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(i)
        label_names.append(file)
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        print(f"✅ Trained with {len(faces)} saved faces.")
        return True, label_names
    else:
        print("⚠ No training data found in known_faces/")
        return False, []

trained, label_names = train_recognizer()

cap = cv2.VideoCapture(0)
print("🎥 Recognition mode ON — press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        face_roi = gray[y:y+h, x:x+w]
        text = "Unknown"

        if trained:
            try:
                label, confidence = recognizer.predict(face_roi)
                name = label_names[label]
                text = f"{name} ({int(confidence)})"
            except Exception:
                text = "Error"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()