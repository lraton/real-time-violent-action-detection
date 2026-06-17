---
sidebar_position: 5
title: Data Preprocessing
description: How raw video feeds are converted into keypoint datasets for LSTM training
keywords: [data preprocessing, keypoint extraction, torso normalization, video processing, movement score, spectator filtering, npz dataset, dataset preparation]
---

# Data Preprocessing

Before training the LSTM model, raw videos must be converted into structured, normalized keypoint sequences. This is handled by the script `src/preprocess_videos.py`.

---

## Script Purpose

The script automates the extraction of human pose keypoints from video datasets using YOLOv11-Pose and saves them as NumPy (`.npz`) file packets ready for LSTM-based action classification.

```
┌─────────────┐     ┌─────────────────────┐     ┌───────────────────────┐     ┌─────────────┐
│  Raw Video  │ ──> │ YOLOv11-Pose Track  │ ──> │ Normalize (To Torso)  │ ──> │ Save .npz   │
│  (.mp4/.avi)│     │  (Extract Keypoints)│     │  & Filter Movements   │     │  Dataset    │
└─────────────┘     └─────────────────────┘     └───────────────────────┘     └─────────────┘
```

---

## Code Workflow

The preprocessing pipeline follows these steps:

### 1. File Scanner & Label Assignment
* The script scans directories for video files (`.mp4`, `.avi`, `.mov`).
* It automatically assigns binary labels based on the folder path:
  - Folders with **"violent"** in their path: Assigned base label `1` (Violent).
  - Other folders: Assigned label `0` (Non-violent/Safe).

### 2. Multi-Person Tracking
* It processes each video frame-by-frame and feeds them into the YOLOv11 pose estimation model (`yolo11n-pose.pt`).
* It initializes tracking using the `botsort.yaml` config. This allows the script to follow the keypoints of a specific person across multiple frames, separating their movement from others in the background.

### 3. Torso-Relative Normalization
To make the coordinate sequence scale-invariant (e.g. when a person walks closer to or further from the camera), keypoints are normalized relative to the torso size:
1. Calculates the center point between the left shoulder (Index 5) and the right shoulder (Index 6).
2. Calculates the Euclidean distance between the two shoulders, representing the **shoulder width**.
3. Centers all 17 keypoints by subtracting the shoulder center, and scales them by dividing by the shoulder width:
   `Normalized Point = (Point - Torso Center) / Shoulder Width`
4. Re-appends the detection confidence score to each keypoint to create a triplet array of shape `(frames, 17, 3)`.

### 4. Logic Filtering & Movement Score
Not all people in a "violent" video are acting violently. Spectators, for instance, stand still while violence occurs. To filter out passive tracks:

#### For Violent Videos:
* Discards tracks that contain less than 30 frames of data.
* Calculates a **Movement Score**: The average frame-to-frame change (Euclidean distance magnitude) of keypoint vectors.
  - **Movement Score Less than 1.5**: Reclassified as **Spectator (Label 0)**.
  - **Movement Score Greater than or equal to 1.5**: Kept as **Violent (Label 1)**.

#### For Non-Violent Videos:
* Discards short tracks with 15 frames or less.
* Saves all remaining tracks with **Label 0**.

### 5. Saving Keypoints
Processed sequences are saved in the `../datasets/lstm_dataset/` directory. Files use the naming convention:
```
{violent|nonviolent}_video{filename}_{cam}_person{id}.npz
```
Each `.npz` file contains two keys:
- `data`: Float32 NumPy array of shape `(frames, 17, 3)` containing keypoints `(x, y, conf)`.
- `label`: Integer scalar (`0` or `1`).

---

## Parallel Execution

Extracting pose keypoints from high-resolution video databases is computationally expensive. The script uses Python's `multiprocessing.Pool` to spawn multiple camera folder workers in parallel.

The number of active worker processes is set dynamically:
```python
num_workers = min(len(video_dirs), os.cpu_count() // 2)
```

---

## Execution Command

Ensure the pre-trained pose model `../models/yolo11n-pose.pt` is present, and run the preprocessing script from the `src/` directory:

```bash
python preprocess_videos.py
```
This will populate the `datasets/lstm_dataset/` directory with `.npz` files.
