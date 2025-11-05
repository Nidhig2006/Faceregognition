import cv2
import os

# Same folder as before
KNOWN_DIR = "known_faces"
os.makedirs(KNOWN_DIR, exist_ok=True)

# Haar cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print("🎥 Camera on — press 'S' to capture a face, 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangle for preview (optional)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break  # only draw the first face

    cv2.imshow("Capture Mode - press 'S' to save", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]                 # save as grayscale (same as training)
            next_idx = len([f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]) + 1
            filename = os.path.join(KNOWN_DIR, f"user_{next_idx}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"💾 Saved: {filename}")
        else:
            print("⚠ No face detected — try again.")

cap.release()
cv2.destroyAllWindows()