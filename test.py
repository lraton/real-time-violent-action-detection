import cv2
import numpy as np
import torch
from ultralytics import YOLO
from openpose import pyopenpose as op
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize YOLO for object detection
model = YOLO('yolov8s.pt')  # Or custom trained model

# Initialize OpenPose parameters
params = dict()
params["model_folder"] = "models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Define the LSTM model for motion classification
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, 50)),
    LSTM(32),
    Dense(2, activation='softmax')  # 2 classes: stabbing, non-stabbing
])
lstm_model.load_weights('lstm_motion_classifier.h5')  # Pre-trained model

# Function to extract arm keypoints

def extract_arm_vectors(keypoints):
    if keypoints is None or len(keypoints.shape) != 3:
        return None

    arms = []
    for person in keypoints:
        arm = person[[2, 3, 4, 5, 6, 7], :2].flatten()  # wrists, elbows, shoulders
        arms.append(arm)

    return np.array(arms).flatten() if arms else None

# Video processing
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy()
    persons = [det for det in detections if int(det[5]) == 0]  # Class 0: Person
    knives = [det for det in detections if int(det[5]) == 43]  # Class 43: Knife (if trained)

    # OpenPose detection
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    arm_vector = extract_arm_vectors(datum.poseKeypoints)
    if arm_vector is not None:
        frame_buffer.append(arm_vector)

    if len(frame_buffer) == 10:
        input_data = np.array(frame_buffer).reshape((1, 10, -1))
        prediction = lstm_model.predict(input_data)
        stabbing_prob = prediction[0][1]

        # Threshold for suspicious action
        if knives and stabbing_prob > 0.8:
            cv2.putText(frame, "ALERT: Potential stabbing!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow('Real-Time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()