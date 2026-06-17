---
sidebar_position: 3
title: Getting Started
description: Installation, prerequisites, and model setup for the project
keywords: [installation guide, setup, CUDA, TensorFlow, PyTorch, YOLOv11-Pose, environment setup, python, virtual environment]
---

# Getting Started

Follow this guide to set up the environment, install the necessary dependencies, and prepare the models for inference.

---

## Prerequisites

Before setting up the project, ensure you meet the following requirements:

### Operating System
* **Windows 10/11** (Tested with Tkinter GUI support).
* **Linux / WSL** (Recommended for CUDA/GPU-accelerated training and inference).

### Software Requirements
* **Python 3.8+** (Python 3.9 or 3.10 is highly recommended).
* **CUDA Toolkit & cuDNN** (If you intend to run real-time inference or training with GPU acceleration).
  * Check your CUDA version by running:
    ```bash
    nvidia-smi
    ```
  * Match your TensorFlow and PyTorch installations to your installed CUDA version.

---

## Step 1: Clone the Repository

Clone the project repository from GitHub and navigate to the project directory:

```bash
git clone https://github.com/lraton/real-time-violent-action-detection.git
cd real-time-violent-action-detection
```

---

## Step 2: Install Dependencies

It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts.

### Create a Virtual Environment

#### On Windows:
```powershell
python -m venv src/.venv
src\.venv\Scripts\activate
```

#### On Linux / WSL:
```bash
python3 -m venv src/.venv
source src/.venv/bin/activate
```

### Install Required Packages

Navigate to the `src/` directory where the dependency manifest is located, and install the packages:

```bash
cd src
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> The `requirements.txt` installs OpenCV, PyTorch (Ultralytics dependency), TensorFlow, Tkinter dependencies, and helper libraries.
> If you have a GPU, ensure PyTorch and TensorFlow are installed with CUDA support.
> For example:
> * **PyTorch (CUDA 12.1)**: `pip install torch torchvision --index-url https://download.pytorch.org/PR/whl/cu121`
> * **TensorFlow (CUDA Support)**: Refer to the official TensorFlow installation guide for matching versions.

---

## Step 3: Set Up Model Weights

The system relies on three separate pre-trained networks. Ensure the following files are placed in their respective folders in the project root:

```
real-time-violent-action-detection/
└── models/
    ├── yolo11n-pose.pt               <-- YOLOv11 Pose estimation model
    ├── lstm_violence_detector_v8.keras <-- LSTM violence action classifier
    └── knife/
        └── run2/
            └── weights/
                └── best.pt           <-- Fine-tuned knife detection model
```

* **YOLOv11-Pose Model**: Automatically downloaded on first run or can be manually placed in the `models/` directory as `yolo11n-pose.pt`.
* **Knife Detection Model**: The fine-tuned YOLO model trained specifically on knives. It should be placed in `models/knife/run2/weights/best.pt`.
* **LSTM Classifier Model**: The recurrent sequence classifier trained on the keypoint dataset. It should be placed in `models/lstm_violence_detector_v8.keras`.
