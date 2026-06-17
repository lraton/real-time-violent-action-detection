---
sidebar_position: 1
title: Introduction & Overview
description: Overview of the Real-time Detection System for Suspicious Stabbing Movements
keywords: [violence detection, stabbing detection, artificial intelligence, computer vision, real-time security, pose estimation, YOLOv11]
---

# Real-time Detection System for Suspicious Stabbing Movements

An advanced real-time surveillance system designed to detect violent interactions and potentially dangerous stabbing movements using Computer Vision and Deep Learning.

```
                  ┌────────────────┐
                  │  Video Stream  │
                  └───────┬────────┘
                          ▼
            ┌───────────────────────────┐
            │   Multi-Stage AI Model    │
            │  (YOLOv11 + Pose + LSTM)  │
            └─────────────┬─────────────┘
                          ▼
             /─────────────────────────\
            <   Suspicious Movement?   >
             \─────────────────────────/
                │ Yes             │ No
                ▼                 ▼
         ┌─────────────┐   ┌─────────────┐
         │ Save Face & │   │   Render    │
         │ Alert Logs  │   │ Normal Feed │
         └─────────────┘   └─────────────┘
```

## Overview

This project implements a state-of-the-art multi-stage pipeline to identify violent intent in real-time video streams. By combining object detection, pose estimation, and temporal sequence analysis, the system can distinguish between normal daily activities and suspicious or violent stabbing motions. 

Unlike simple frame-by-frame classifiers, our system uses a **Long Short-Term Memory (LSTM)** neural network to evaluate keypoint movement vectors over a temporal buffer, allowing it to understand the *context* of movements over time.

---

## Key Features

- **Real-Time Analysis**: Process local video feeds or camera streams with minimal latency.
- **Dual Stream Detection**:
  - **Object Detection (YOLOv11)**: Specifically fine-tuned to identify weapons (e.g. knives) with high precision.
  - **Pose Estimation (YOLOv11-Pose)**: Tracks skeletal keypoints (shoulders, elbows, wrists) of multiple persons in the frame.
- **Temporal Sequence Modeling**: Uses a Bidirectional LSTM network tracking a buffer of up to 150 frames of normalized keypoints to analyze arm swing speeds and stabbing motion dynamics.
- **Smart Frame Skipping**: Implements dynamic frame skipping (processing 1 frame every $N$) to optimize performance on CPU/GPU hardware while maintaining the underlying keypoint sequence history.
- **Automatic Alert System**:
  - Automatically crops and saves the face of the suspicious/violent person in a dedicated folder (`suspect/`).
  - Records detailed timestamped logs (`logs/`) with person IDs, action labels, and detection confidence values.
- **Interactive UI**: Fully featured graphical interface built with Tkinter, featuring live feed rendering, dynamic frame skipping adjustment, and quick folder shortcuts.

---

## Tech Stack

The application is built on top of a modern machine learning and computer vision stack:

- **Programming Language**: Python 3.8+
- **Computer Vision**: 
  - [OpenCV](https://opencv.org/) for image manipulation and video capture.
  - [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for object detection (weapons) and human pose tracking.
- **Deep Learning**:
  - [TensorFlow / Keras](https://tensorflow.org/) for loading and running the temporal Bidirectional LSTM network.
- **Data Engineering**:
  - [Roboflow](https://roboflow.com/) for dataset preparation and annotation.
  - [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) for vector normalization and math operations.
- **Graphical User Interface**:
  - [Tkinter](https://docs.python.org/3/library/tkinter.html) for a lightweight, cross-platform dashboard.

---

## Demo Videos

You can view the system in action on different test cases (both violent actions and normal daily tasks) on YouTube:

🎥 [Suspicious Stabbing Movements Detection Playlist](https://www.youtube.com/playlist?list=PLo1f0U_Wr1t2QE8hyGbKqB8WCAeb5DOUH)

The playlist shows several scenarios:
1. **Aggressive Stabbing Detection**: Red warning boxes and alerts triggered immediately.
2. **Normal Interactions**: Green boxes highlighting daily hand movements and normal walking, proving low false-positive rates.
3. **Weapon Detection**: Blue bounding boxes highlighting visible knives.
