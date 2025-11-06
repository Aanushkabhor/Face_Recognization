import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np


def read_dataset(dataset_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    id_to_name: Dict[int, str] = {}

    if not os.path.isdir(dataset_dir):
        print(f"Error: dataset directory not found: {dataset_dir}")
        sys.exit(1)

    person_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if not person_names:
        print("Error: dataset is empty. Run 1_collect_faces.py first.")
        sys.exit(1)

    for person_id, person_name in enumerate(person_names):
        id_to_name[person_id] = person_name
        person_path = os.path.join(dataset_dir, person_name)
        for filename in os.listdir(person_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            images.append(image)
            labels.append(person_id)

    if not images:
        print("Error: no images found in dataset.")
        sys.exit(1)

    return images, labels, id_to_name


def train_and_save_lbph(dataset_dir: str, models_dir: str) -> None:
    images, labels, id_to_name = read_dataset(dataset_dir)

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lbph_model.yml")
    labels_path = os.path.join(models_dir, "labels.txt")

    recognizer.save(model_path)

    with open(labels_path, "w", encoding="utf-8") as f:
        for person_id in sorted(id_to_name.keys()):
            f.write(f"{person_id},{id_to_name[person_id]}\n")

    print(f"Saved model to: {model_path}")
    print(f"Saved labels to: {labels_path}")


if __name__ == "__main__":
    train_and_save_lbph(dataset_dir="dataset", models_dir="models")


