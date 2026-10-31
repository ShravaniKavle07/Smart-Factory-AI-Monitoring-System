# 🏭 Smart Factory AI Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLO-11-8A2BE2?logo=ultralytics&logoColor=white" alt="YOLO11"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

## 📌 Overview

The **Smart Factory AI Monitoring System** is a computer vision project designed to support industrial safety monitoring using object detection. It uses a custom-trained YOLO11 model to identify people, personal protective equipment (PPE), machinery, and vehicles in images and videos.

The project explores how AI can help monitor workplace safety and support smarter manufacturing environments.

## 🎯 Objectives

- 👷 Detect people in industrial and construction-site images.
- 🪖 Identify hard hats and safety-related equipment.
- 🦺 Detect safety vests and missing-PPE categories.
- 🚜 Identify machinery and vehicles.
- 📹 Process images and videos using computer vision.
- 📊 Present detection results and object counts.
- 🚨 Explore PPE monitoring and safety-alert reporting.

## ✨ Features

- 🔍 **Object Detection:** Detects multiple object categories using YOLO11.
- 🖼️ **Image Inference:** Runs predictions on images and visualizes detected objects.
- 🎥 **Video Inference:** Processes video frames and saves annotated output.
- 🦺 **PPE Monitoring:** Supports analysis of hard hats, safety vests, and missing-PPE detections.
- 📈 **Detection Counts:** Summarizes detected object categories.
- 🖥️ **Dashboard:** A Streamlit interface can be used to present results once implemented.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11 (Ultralytics) | Object detection |
| PyTorch | Deep learning framework |
| OpenCV | Image and video processing |
| NumPy | Numerical operations |
| Pandas | Data handling and reporting |
| Streamlit | Interactive dashboard, if implemented |
| Jupyter Notebook | Experimentation and model development |

## 🏷️ Detection Classes

The dataset contains the following 10 classes:

| ID | Class |
|---:|---|
| 0 | Hardhat |
| 1 | Mask |
| 2 | NO-Hardhat |
| 3 | NO-Mask |
| 4 | NO-Safety Vest |
| 5 | Person |
| 6 | Safety Cone |
| 7 | Safety Vest |
| 8 | Machinery |
| 9 | Vehicle |

## 📂 Project Structure

```text
Smart_Factory_ai_monitoring_system/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── models/
│   └── best.pt
│
├── notebooks/
│   ├── code.ipynb
│   └── dashboard.ipynb
│
├── output/
│   ├── images/
│   └── videos/
│
├── app.py
├── requirements.txt
└── README.md
```


## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Smart_Factory_ai_monitoring_system.git
cd Smart_Factory_ai_monitoring_system
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

If your `requirements.txt` is ready:

```bash
pip install -r requirements.txt
```

Otherwise, install the core packages:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib streamlit jupyter
```

## 🧠 Model Training

The model can be trained using the Ultralytics YOLO11 framework. The following example uses lightweight settings for CPU-based training.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="../dataset/data.yaml",
    epochs=5,
    imgsz=416,
    batch=4,
    workers=2,
    device="cpu",
    project="../runs",
    name="smart_factory"
)
```

Run this code from the `notebooks` directory. Adjust the dataset path if your notebook is located elsewhere.

After training, locate the generated `best.pt` file and copy it into the `models/` folder if you want to load it from there.

## 🔎 Run Predictions

Example for image inference from a notebook:

```python
from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("../models/best.pt")

image_path = Path("../sample_images/factory.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

results = model.predict(source=image, save=False)
annotated_image = results[0].plot()

cv2.imwrite("../output/images/predicted_image.jpg", annotated_image)
```

Update the image path to an image that exists in your project. Create the `sample_images/` and output directories if needed.

## 📊 Evaluation

Model performance should be evaluated using suitable object-detection metrics, including:

- **Precision:** How many predicted detections are correct.
- **Recall:** How many relevant objects are detected.
- **mAP50:** Mean Average Precision at an IoU threshold of 0.50.
- **mAP50-95:** Mean Average Precision across multiple IoU thresholds.
- **Inference time:** Time required to process an image or video frame.

*Add your actual evaluation results after training and validation. Do not report unmeasured accuracy or performance values.*

## ⚠️ Limitations

- PPE monitoring based only on object counts does not confirm whether each individual worker is compliant.
- Detection performance depends on image quality, lighting, object size, and dataset coverage.
- A model trained on construction-site images may not generalize directly to every factory environment.
- CPU-based training and inference can be slower than GPU-based execution.
- The system is a prototype and should not replace established workplace safety procedures.

## 🚀 Future Enhancements

- 🔗 Track individual workers across video frames.
- 🚨 Generate more reliable PPE violation alerts.
- 📉 Add detection metrics and safety statistics to a dashboard.
- 📹 Support real-time camera feeds.
- ⚡ Explore edge-AI deployment for industrial environments.
- 🏭 Validate the model using images from actual factory settings.

## 📚 Dataset and Attribution

Dataset: [Construction Site Safety — Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)

The dataset page identifies the dataset as licensed under **CC BY 4.0**. Review the dataset's current license and attribution requirements before redistributing images or annotations. Credit the original dataset creators when required.

## 👩‍💻 Author

**Shravani Kavle**

- GitHub: [@ShravaniKavle07](https://github.com/ShravaniKavle07)

---

<p align="center">
  ⭐ If you find this project useful, consider giving the repository a star!
</p>


