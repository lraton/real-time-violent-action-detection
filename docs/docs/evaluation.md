---
sidebar_position: 7
title: Evaluation & Performance
description: Performance metrics, evaluation framework, model accuracy, and flicker rate stability
keywords: [evaluation metrics, confusion matrix, ROC-AUC, PR-AUC, flicker rate, YOLOv11 evaluation, LSTM accuracy, Filippo Notari thesis]
---

# Evaluation & Performance

This page outlines the evaluation metrics and experimental results obtained during the system validation. All numbers are aligned with the research findings documented in the academic thesis.

---

## 1. YOLOv11 Knife Detection (Object Detection)

The custom-trained knife detection model was evaluated on a test split of **704 images** containing **779 knife instances**.

### Performance Metrics

| Class | Precision (P) | Recall (R) | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Knife (All)** | **0.890** | **0.825** | **0.856** | **0.910** | **0.612** |

* **Precision (0.890)**: Demonstrates high reliability, minimizing false weapon alerts (crucial for security monitoring to prevent unnecessary panic).
* **mAP@0.5 (0.910)**: Indicates exceptional spatial overlap accuracy under loose boundary criteria.
* **mAP@0.5:0.95 (0.612)**: Shows expected regression limitations under extremely strict boundary alignments, typical for small objects in motion.

### Confusion Matrix (Object Detection)
* **True Positives (Knife correctly detected)**: 685
* **False Positives (Background marked as Knife)**: 151
* **False Negatives (Knife missed by model)**: 94
* *Note: True Negatives are not tracked, which is standard for single-class object detectors.*

---

## 2. YOLOv11-Pose Estimation Benchmarks

To establish baseline skeletal accuracy, the pose estimator was benchmarked against the standard **COCO val2017** dataset.

### Performance Metrics

| Task | Precision (P) | Recall (R) | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Skeletal Pose** | **0.849** | **0.760** | **0.802** | **0.810** | **0.511** |

The model provides stable, occlusion-resistant tracking of key joints (shoulders, elbows, wrists) even during physical contact or overlapping profiles.

---

## 3. Temporal Bi-LSTM Classifier

The temporal sequence classifier was evaluated using a strictly isolated Test Set composed of **183 skeletal sequences** derived from 55 source videos, ensuring zero data leakage.

### Classification Metrics

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Non-Violent (Safe)** | 0.65 | 0.79 | 0.71 |
| **Violent (Danger)** | **0.87** | **0.76** | **0.81** |
| **Overall Accuracy** | **0.77 (77%)** | | |

* **High Violent Precision (0.87)**: Ensures that when the system logs a violent interaction or sounds a warning, there is an 87% chance an actual threat is occurring.
* **Balanced Violent Recall (0.76)**: Correctly flags three-quarters of all violent sequence events.

### Confusion Matrix (Bi-LSTM per Video)

| Predicted \ Actual | True Violent | True Non-Violent |
| :--- | :---: | :---: |
| **Predicted Violent** | **26** (True Positive) | **4** (False Positive) |
| **Predicted Non-Violent** | **8** (False Negative) | **15** (True Negative) |

* **Violent Detection Rate**: Correctly classified **76.4%** of the violent video streams (26 out of 34).
* **Safe Detection Rate**: Correctly classified **78.9%** of the non-violent control streams (15 out of 19).

### ROC-AUC Score
* **ROC-AUC**: **0.6749**
The ROC curve demonstrates performance significantly above random chance (0.5), showing that the Bi-LSTM model successfully isolates the kinematics of stabbing gestures.

---

## 4. Signal Stability (Flicker Rate)

The **Flicker Rate** measures the temporal stability of model predictions across consecutive frames to prevent erratic visual alert toggling.

* **Flicker Rate**: **0.0202 (2.02%)**

A rate of 2.02% is **excellent**. It indicates that once a person is classified under a threat status, the alert label remains solid and constant, preventing visual "flicker" on the operator screen. This stability is achieved by:
1. Posing a temporal threshold constraint of **30 consecutive frames** before starting inferences.
2. Implementing a fallback rendering mechanism that repeats the last processed frame during frame skips instead of switching back to unannotated feeds.
