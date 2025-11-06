import os
import sys
from typing import Dict

import cv2


def load_labels(labels_path: str) -> Dict[int, str]:
    if not os.path.isfile(labels_path):
        print("Error: labels file not found. Run 2_train_lbph.py first.")
        sys.exit(1)
    id_to_name: Dict[int, str] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            person_id_str, name = line.split(",", 1)
            id_to_name[int(person_id_str)] = name
    return id_to_name


def load_model(model_path: str) -> cv2.face_LBPHFaceRecognizer:
    if not os.path.isfile(model_path):
        print("Error: model file not found. Run 2_train_lbph.py first.")
        sys.exit(1)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer


def load_face_detector(cascade_path: str) -> cv2.CascadeClassifier:
    if not os.path.isfile(cascade_path):
        print(f"Error: Haar cascade not found at: {cascade_path}")
        sys.exit(1)
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("Error: Failed to load Haar cascade file.")
        sys.exit(1)
    return detector


def main() -> None:
    model_path = os.path.join("models", "lbph_model.yml")
    labels_path = os.path.join("models", "labels.txt")
    cascade_path = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")

    id_to_name = load_labels(labels_path)
    recognizer = load_model(model_path)
    face_detector = load_face_detector(cascade_path)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Cannot access the webcam.")
        sys.exit(1)

    print("Starting real-time recognition. Press 'q' to quit.")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Warning: Failed to read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                roi_gray = gray[y : y + h, x : x + w]
                predicted_id, confidence = recognizer.predict(roi_gray)
                name = id_to_name.get(predicted_id, "Unknown")
                label = f"{name} ({confidence:.1f})"
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            cv2.imshow("LBPH Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


