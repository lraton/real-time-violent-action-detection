---
sidebar_position: 6
title: Model Training
description: Detailed guide to training the YOLO detection models and the temporal LSTM classifier
keywords: [model training, YOLOv11 training, LSTM architecture, Bidirectional LSTM, learning rate, neural network optimization, checkpoints, class weighting]
---

# Model Training

The system uses three models working together. This guide explains how each model is configured, trained, and saved.

---

## 1. YOLOv11 Knife Detector

To identify knives and weapons in video frames with high precision, we fine-tune a pre-trained YOLOv11 network.

### Dataset Preparation
* The training dataset is hosted and prepared on **Roboflow**. It is labeled with bounding boxes around knives and similar stabbing weapons.
* The dataset should be exported in the YOLO format, generating a `data.yaml` configuration pointing to image splits.

### Training Command
Run the following CLI command to fine-tune the model:

```bash
yolo detect train data="knife-1/data.yaml" model=yolo11n.pt epochs=181 imgsz=640
```

* **Model**: Starts from the lightweight, high-speed `yolo11n.pt` base weights.
* **Epochs**: Trained for `181` epochs.
* **Image Size**: Resized to `640x640` pixels.
* **Result**: Saves the best weights under `runs/detect/train/weights/best.pt`, which should be placed in `models/knife/run2/weights/best.pt`.

---

## 2. YOLOv11 Pose Estimator

The pose estimation model tracks coordinates of human body joints. You can either use pre-trained weights (`yolo11n-pose.pt` trained on the COCO dataset) or fine-tune them.

To train the model on a custom pose dataset:

```bash
yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
```

---

## 3. Bidirectional LSTM Classifier

The LSTM network is responsible for analyzing the keypoint sequences extracted by `preprocess_videos.py` and determining if they represent a violent stabbing action.

### Script Config
The training parameters are configured at the top of `src/train_lstm.py`:
* `MAX_FRAMES = 150`: The target length for keypoint sequences. Shorter sequences are padded; longer ones are truncated.
* `MASK_VALUE = -999.0`: Floating-point placeholder value used to represent empty frames.
* `BATCH_SIZE = 32`
* `EPOCHS = 100`

### Model Architecture
The Keras sequential architecture is structured as follows:

| Layer | Type | Specifications | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Masking** | `mask_value=-999.0` | Skips padded frames to focus computation on active movements. |
| **2** | **Bidirectional LSTM** | 64 units, returns sequences | Evaluates coordinate movements forwards and backwards in time. |
| **3** | **Dropout** | Rate: `0.3` | Mitigates overfitting. |
| **4** | **Bidirectional LSTM** | 32 units | Refines temporal context representations. |
| **5** | **Dropout** | Rate: `0.3` | Mitigates overfitting. |
| **6** | **Batch Normalization** | Default | Stabilizes gradients and accelerates training. |
| **7** | **Dense** | 32 units, Activation: `ReLU` | Dense classifier layer. |
| **8** | **Dropout** | Rate: `0.2` | Mitigates overfitting. |
| **9** | **Dense (Output)** | 1 unit, Activation: `Sigmoid` | Outputs probability value from `0.0` (Safe) to `1.0` (Violent). |

### Data Preparation & Training Flow
1. **Load npz Data**: Automatically scans `datasets/lstm_dataset/` for `.npz` files, loading input arrays (`X`) and targets (`y`).
2. **Padding**: Pads sequence inputs using Keras `pad_sequences(..., padding='post', value=-999.0)`.
3. **Data Split**: Splits loaded samples into `80%` Training and `20%` Validation sets, maintaining class distributions.
4. **Class Weighting**: Since databases may have unequal counts of violent versus non-violent samples, the script computes class weights to prevent model bias:
   ```python
   class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
   ```
5. **Optimization**: Compiled using the `Adam` optimizer and `binary_crossentropy` loss.

### Training Callbacks
* **Model Checkpoint**: Automatically monitors `val_loss` and saves the best iteration to `models/lstm_violence_detector_tmp.keras`.
* **Early Stopping**: Prevents training when validation loss stops improving after `10` epochs.
* **Reduce Learning Rate on Plateau**: Multiplies the learning rate by `0.5` if validation loss stagnates for `3` consecutive epochs.

---

## Running the Training Script

Execute the training script from the `src/` directory:

```bash
python train_lstm.py
```
After training completes, retrieve the output file at `models/lstm_violence_detector_tmp.keras` and rename it appropriately for integration with the main inference loop.
