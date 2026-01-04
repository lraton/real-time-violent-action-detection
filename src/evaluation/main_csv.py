import cv2
import os
from violence_detection_system_csv import ViolenceDetectionSystem

VIDEOS_DIR = "dataset_eval/"
FRAME_SKIP = 2

CLASS_MAP = {"safe": 0, "violence": 1, "weapon": 2}


def evaluate():
    app = ViolenceDetectionSystem(knife_model_path="../../models/knife/run2/weights/best.pt",
                                  pose_model_path="../../models/yolo11n-pose.pt",
                                  lstm_model_path="../../models/lstm_violence_detector_v8.keras")

    for category, true_class in CLASS_MAP.items():
        category_path = os.path.join(VIDEOS_DIR, category)
        if not os.path.exists(category_path):
            continue

        for video_name in os.listdir(category_path):
            if not video_name.endswith(".mp4"):
                continue

            video_path = os.path.join(category_path, video_name)
            #print(f"Processing {category}/{video_name}")

            cap = cv2.VideoCapture(video_path)
            app.last_video_id = None  # Reset video state

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                app.process_frame(frame, FRAME_SKIP, video_name, true_class)
                frame_id += 1

            cap.release()

    print("Evaluation completed.")


if __name__ == "__main__":
    evaluate()
