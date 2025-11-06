# Face Recognition System (LBPH)

## Required Files and Libraries

To run the Face Recognition System, several essential files and Python libraries are required for the project to function smoothly in Visual Studio Code (VS Code). The project mainly consists of four Python scripts — `1_collect_faces.py`, `2_train_lbph.py`, `3_recognize_realtime.py`, and `4_recognize_image.py` — which handle different stages such as image collection, model training, real-time recognition, and image testing. Additionally, a `haarcascades` folder is needed to store the Haar Cascade classifier file (`haarcascade_frontalface_default.xml`), which is used for face detection. The `dataset` folder stores all captured face images of individuals, while the `models` folder contains the trained LBPH (Local Binary Pattern Histogram) model file (`lbph_model.yml`) and the label file (`labels.txt`) after training.

The required Python libraries for this project include OpenCV, OpenCV-contrib, NumPy, Imutils, and optionally Streamlit for the web-based interface.

- **OpenCV (`cv2`)**: face detection, image processing, webcam operations.
- **OpenCV-contrib**: LBPH face recognizer module for training and recognizing faces.
- **NumPy**: efficient numerical/array operations.
- **Imutils**: simplified image resizing and rotation utilities.
- **Streamlit (optional)**: simple interactive web interface.

Install them (preferably inside a virtual environment) with:

```bash
pip install -r requirements.txt
```

Or explicitly:

```bash
pip install opencv-python opencv-contrib-python numpy imutils streamlit
```

## Project Structure

```
Face Recognization/
├─ haarcascades/
│  └─ haarcascade_frontalface_default.xml   # put file here
├─ dataset/                                 # captured faces per person
├─ models/
│  ├─ lbph_model.yml                        # generated after training
│  └─ labels.txt                            # generated after training
├─ 1_collect_faces.py
├─ 2_train_lbph.py
├─ 3_recognize_realtime.py
└─ 4_recognize_image.py
```

Download the cascade file from OpenCV and place it at `haarcascades/haarcascade_frontalface_default.xml`.

## Quickstart

1) Create and activate a virtual environment (Windows PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Collect faces (press `q` to quit):

```bash
python 1_collect_faces.py --person "Your Name"
```

3) Train LBPH model:

```bash
python 2_train_lbph.py
```

4) Real-time recognition (press `q` to quit):

```bash
python 3_recognize_realtime.py
```

5) Recognize from an image file:

```bash
python 4_recognize_image.py --image path\to\image.jpg
```

## Notes

- Ensure your webcam is accessible.
- Good lighting and front-facing images improve accuracy.
- Re-run training after adding new people to the dataset.
