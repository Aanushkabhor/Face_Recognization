import argparse
import os
import sys
from datetime import datetime

import cv2


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_face_detector(cascade_path: str) -> cv2.CascadeClassifier:
    if not os.path.isfile(cascade_path):
        print(f"Error: Haar cascade not found at: {cascade_path}")
        print("Download 'haarcascade_frontalface_default.xml' and place it in the 'haarcascades' folder.")
        sys.exit(1)
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        print("Error: Failed to load Haar cascade file.")
        sys.exit(1)
    return detector


def collect_faces(person_name: str, output_dir: str, max_images: int, camera_index: int) -> None:
    ensure_directory(output_dir)

    cascade_path = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
    face_detector = load_face_detector(cascade_path)

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print("Error: Cannot access the webcam.")
        sys.exit(1)

    person_dir = os.path.join("dataset", person_name)
    ensure_directory(person_dir)

    print("Starting capture. Press 'q' to quit.")
    saved_count = 0

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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(person_dir, f"{person_name}_{timestamp}.jpg")
                cv2.imwrite(filename, roi_gray)
                saved_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Saved: {saved_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Collecting Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if saved_count >= max_images:
                print(f"Reached target of {max_images} images.")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved_count} images to: {person_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect face images for a person using webcam.")
    parser.add_argument("--person", required=True, help="Person's name (used as dataset subfolder)")
    parser.add_argument("--max", type=int, default=50, help="Max number of images to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_directory("dataset")
    collect_faces(person_name=args.person, output_dir="dataset", max_images=args.max, camera_index=args.camera)


