---
sidebar_position: 9
title: Academic Thesis
description: Academic thesis summary and PDF document download for Filippo Notari's research
keywords: [academic thesis, research paper, Filippo Notari, Francesco Santini, Università degli Studi di Perugia, YOLO, LSTM, violent action detection]
---

# Academic Thesis

This page presents the academic foundation of the **Real-time Violent Action and Stabbing Detection System**. The thesis was researched and written by **Filippo Notari** under the supervision of **Prof. Francesco Santini** at the **Università degli Studi di Perugia** (Department of Mathematics and Computer Science).

---

## Document Metadata

* **Title**: *Sviluppo di un sistema di rilevamento in tempo reale di azioni violente tramite YOLO e LSTM* (Development of a Real-Time Violent Action Detection System Using YOLO and LSTM)
* **Author (Laureando)**: Filippo Notari
* **Advisor (Relatore)**: Prof. Francesco Santini
* **Institution**: Università degli Studi di Perugia (Dipartimento di Matematica e Informatica)
* **Degree Course**: Tesi Triennale in Informatica (Bachelor's Degree in Computer Science)
* **Academic Year**: 2024-2025
* **Language**: Italian (Tesi in Italiano)

---

import useBaseUrl from '@docusaurus/useBaseUrl';

## Document Download & Viewer

To read the full research paper, you can download the PDF directly or use the embedded document viewer below:

📥 <a href={useBaseUrl('/thesis.pdf')} className="button button--secondary">Download the Full Thesis (PDF)</a>

### Embedded PDF Viewer

If you have placed your thesis PDF file at `docs/static/thesis.pdf`, it will render directly inside the browser pane below:

<object
  data={useBaseUrl('/thesis.pdf')}
  type="application/pdf"
  width="100%"
  height="800px"
  style={{ border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '4px' }}
>
  <div style={{ padding: '2rem', textAlign: 'center', backgroundColor: 'var(--ifm-color-emphasis-100)', borderRadius: '4px' }}>
    <p>Your browser does not support embedding PDF files inline.</p>
    <a href={useBaseUrl('/thesis.pdf')} className="button button--primary">Download Thesis PDF</a>
  </div>
</object>

---

## Abstract & Thesis Summary

The research proposes a vision-based surveillance system capable of detecting violent actions—specifically stabbing actions—in real-time video feeds. The architecture leverages a hybrid approach that decouples spatial detection from temporal classification to ensure high accuracy and low latency:

1. **YOLOv11-Pose**: Extracts skeletal pose coordinates (17 keypoints) to represent human biometrics, running without custom fine-tuning.
2. **YOLOv11-Detect**: A custom-trained object detection model fine-tuned on **3,827 images** to localize knives with high confidence.
3. **Bi-LSTM (Bidirectional LSTM)**: A recurrent network trained on a dataset of **17,727 skeletal movement sequences** to analyze coordinates over time and distinguish aggressive stabbing gestures from ordinary daily activities.

---

## Detailed Chapter Structure

### Chapter 1: Introduzione (Introduction)
* **Problem Definition**: Limitations of traditional security networks that rely on passive recording and human operators.
* **Objective**: Automate violent behavior detection to enable proactive security interventions.
* **Challenges**: Varying illumination, camera angles, partial occlusions, and distinguishing aggressive interactions from benign physical contacts (like hugs, sports, or play).

### Chapter 2: Background (State of the Art)
* **Human Activity Recognition (HAR)**: Overview of levels of interaction (Atomic Actions, Human-Human Interaction, Human-Object Interaction, Group Activities) and acquisition techniques (Wearable, Non-wearable, and Vision-based).
* **Model Overviews**: Deep dive into YOLO (Object Detection and YOLO-Pose coordinate regression) and Recurrent Neural Networks (explaining Vanishing/Exploding gradients, LSTM gating mechanics, and Bidirectional LSTM structures).
* **Multi-Object Tracking (MOT)**: Tracking-by-Detection using BoT-SORT to maintain stable person IDs across frames.

### Chapter 3: Metodologia e Implementazione (Methodology & Implementation)
* **Model Choices & Fine-Tuning**:
  - **YOLO11n-pose**: Integrated with stock weights to maintain fast keypoint inference without accuracy loss.
  - **YOLO11n (Knife Detection)**: Fine-tuned on a custom dataset of 3,827 images (Training: 2,865, Val: 258, Test: 704) sourced from Roboflow and Open Images V7. Stopped at epoch 181 with SGD.
  - **Bi-LSTM Classifier**: Trained on extracted skeleton sequences from four video databases: *Automatic Violence Detection*, *Indoor Action*, *Real Life Violence*, and *RWF2000*. To prevent Data Leakage, 55 source videos were completely isolated for the Test Set (183 test sequences). The remaining sequences (17,544) were split 80% train / 20% validation.
* **Torso-Relative Keypoint Normalization**:
  Calculates chest center `C` and shoulder width `W` to translate and scale joints, achieving scale-invariance:
  - `Chest Center (C) = (Left_Shoulder + Right_Shoulder) / 2`
  - `Shoulder Width (W) = Euclidean_Distance(Left_Shoulder, Right_Shoulder)`
  - `Normalized Joint (P'_i) = (Joint_i - C) / W`
* **Temporal Logic**: Accumulates keypoints in a queue (maximum 150 frames, representing 5 seconds of video at 30 FPS). Inference triggers when the sequence length satisfies:
  - `Sequence Length >= 30 / k` (where `k` is the frame skip factor).
* **Software Architecture**: Separates the GUI (Tkinter/OpenCV, calculating live FPS and handling adaptive frame-skipping rendering) from the Backend processing pipeline (`ViolenceDetectionSystem`).

### Chapter 4: Metriche di Valutazione (Evaluation & Results)
* **YOLO-Object (Knives)**: Achieved a Precision of **0.890**, Recall of **0.825**, and F1-Score of **0.856** (mAP@0.5: **0.91**).
* **YOLO-Pose**: Evaluated on COCO val2017, yielding a Precision of **0.849**, Recall of **0.760**, and F1-Score of **0.802** (mAP@0.5: **0.810**).
* **Bi-LSTM Classification**:
  - Overall accuracy: **0.77 (77%)**.
  - Violent class performance: Precision **0.87**, Recall **0.76**, F1-Score **0.81**.
  - ROC-AUC: **0.6749**.
  - **Confusion Matrix**: Out of 34 violent videos, 26 were classified correctly. Out of 19 non-violent videos, 15 were classified correctly.
* **Temporal Stability (Flicker Rate)**: System achieved a highly stable Flicker Rate of **0.0202 (2.02%)**, showing that state transitions are clean and lack jitter.

### Chapter 5: Conclusioni (Conclusions & Future Work)
* **Summary**: Proved the feasibility of lightweight edge deployment by combining YOLO and Bi-LSTM.
* **Future Work**: Expansion to multi-camera cross-tracking, scene context integration (spatial interaction between people and objects), and predictive anticipation of aggression triggers.
