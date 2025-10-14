# Real-time Detection System for Suspicious Stabbing Movements

This project implements a real-time detection system to identify potentially dangerous stabbing movements using advanced computer vision and machine learning techniques.

---

##  Technologies and Tools
- ~~**PyTorch**: ``` pip install torch torchvision numpy scikit-learn opencv-python ``` ~~

- **YOLO (You Only Look Once)**: Object detection with Ultralytics.
    
- **Roboflow**: Data management and preprocessing.
    
- ~~**MediaPipe**: Human pose estimation.~~
    
    - [MediaPipe Google](ai.google.dev/edge/mediapipe/solutions/guide?hl=it)
        
- **Violence Detection Dataset**: Dataset specifically designed for identifying violent actions.
    
    - [Violence Detection Dataset GitHub](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/master)
        

---

## 📝 Workflow Steps

1. **Object Detection (YOLO)**:
    
    - Analyze each video frame to detect instances of:
        
        - `person`
            
        - `knife`
            
    - Associate detected knives with the closest identified hand/person.
        
2. **Pose Estimation (YOLO)**:
    
    - Extract keypoints such as wrists, elbows, and shoulders.
        
    - Calculate and track **motion vectors** of arms and hands.
        
3. **Human Activity Recognition HAR**:
    
    - Use machine learning methods (e.g., LSTM or CNN) to classify motion patterns over time.
        
    - Identify suspicious patterns indicative of stabbing or violent intent.
        
4. **Alert Triggering**:
    
    - Generate a real-time alert if the system detects:
