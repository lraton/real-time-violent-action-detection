import cv2
import numpy as np
import os
from openpose import pyopenpose as op
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Initialize OpenPose
params = dict()
params["model_folder"] = "models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def extract_keypoints(frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    keypoints = datum.poseKeypoints
    if keypoints is None or len(keypoints.shape) != 3:
        return None
    arm_keypoints = keypoints[0][[2,3,4,5,6,7], :2].flatten()
    return arm_keypoints

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            sequence.append(keypoints)
    cap.release()
    return sequence

def create_dataset(base_dir):
    X, y = [], []
    classes = {'non_violent':0, 'violent':1}
    for label in classes:
        label_path = os.path.join(base_dir, label)
        for cam_dir in os.listdir(label_path):
            cam_path = os.path.join(label_path, cam_dir)
            for video_file in os.listdir(cam_path):
                video_path = os.path.join(cam_path, video_file)
                sequence = process_video(video_path)
                if len(sequence) >= 10:
                    for i in range(len(sequence) - 9):
                        X.append(sequence[i:i+10])
                        y.append(classes[label])
    return np.array(X), np.array(y)

# Prepare dataset
X, y = create_dataset(r'C:\Users\notar\Desktop\Programmi\RT-Detection-System-for-Suspicious-Stabbing-Movements')

y_cat = to_categorical(y, num_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, X.shape[2])),
    LSTM(32),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

# Save the model
model.save('lstm_motion_classifier.h5')
