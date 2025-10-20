from ultralytics import YOLO
from natsort import natsorted
import numpy as np
import cv2
import os

# Carica il modello
model_pose = YOLO("yolo11n-pose.pt")

path_to_video = '../video-dataset/violent/cam1/'

# INDICI DEI KEYPOINT NEL FORMATO COCO (usato da YOLO)
# 5: spalla sinistra (left_shoulder)
# 6: spalla destra (right_shoulder)
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

def main():
    for filename in natsorted(os.listdir(path_to_video)):

        people_data = {}  # {person_id: [frames]} dati per ogni persona per video

        # Avvia il tracking sul video 
        results_generator = model_pose.track(
            path_to_video + filename,
            tracker="botsort.yaml",
            #persist=True,
            stream=True,
            verbose=False
        )
        
        print(f"Video {filename}")
        # Processa ogni frame del video
        for frame_result in results_generator:
            if frame_result.boxes.id is not None:

                person_ids = frame_result.boxes.id.cpu().numpy().astype(int)
                keypoints_normalized = frame_result.keypoints.xyn.cpu().numpy()

                # Per ogni persona rilevata nel frame
                for i, person_id in enumerate(person_ids):
                    relative_keypoints = normalize_keypoints_relative_to_torso(i, keypoints_normalized)  # Calcola i keypoint relativi
                    
                    
                    conf = frame_result.keypoints.conf[i].cpu().numpy()  # sposta su CPU e converti in NumPy
                    relative_keypoints_with_conf = np.hstack([relative_keypoints, conf[:, None]])  # Aggiungi la conf come terza colonna

                    if person_id not in people_data:    # Inizializza la lista per la persona se non esiste
                        people_data[person_id] = []
                    people_data[person_id].append(relative_keypoints_with_conf)  # Aggiungi i keypoint relativi alla lista della persona

                    # Salva i keypoint normalizzati nel dataset                    
                    save_keypoints_to_dataset(people_data, filename)

            #video_display(frame_result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print(f"People data collected so far: {len(people_data)} individuals.")
        break

# Funzione per normalizzare i keypoint rispetto al centro del torace
def normalize_keypoints_relative_to_torso(i, keypoints_normalized):
    person_keypoints_xyn = keypoints_normalized[i]

    left_shoulder = person_keypoints_xyn[LEFT_SHOULDER_IDX]
    right_shoulder = person_keypoints_xyn[RIGHT_SHOULDER_IDX]

    if left_shoulder.sum() > 0 and right_shoulder.sum() > 0:
        center_point = (left_shoulder + right_shoulder) / 2
    else:
        center_point = np.array([0.0, 0.0])

    relative_keypoints = person_keypoints_xyn - center_point

    return relative_keypoints

#   Salva i keypoint normalizzati nel dataset
def save_keypoints_to_dataset(people_data, filename):
    save_path='models/violentvideo/'
    os.makedirs(save_path, exist_ok=True)

    for pid, frames in people_data.items():
        if len(frames) == 0:
            continue  # salta persone senza frame

        person_array = np.array(frames, dtype=np.float32)  # (num_frames, 17, 3)
        video_name = os.path.splitext(os.path.basename(filename))[0]

        np.savez(f"{save_path}video{video_name}_person{pid}", data=person_array, label=1)

# Funzione per visualizzare il video con i box e i keypoint annotati
def video_display(frame_result):
    annotated_frame = frame_result.plot()
    display_frame = cv2.resize(
        annotated_frame, (1080, 720), interpolation=cv2.INTER_AREA)
    # Mostra il frame nella finestra ridimensionabile
    cv2.imshow("YOLO Tracking", display_frame)


if __name__ == '__main__':
    main()
