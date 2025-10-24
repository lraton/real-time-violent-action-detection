from ultralytics import YOLO
from natsort import natsorted
import numpy as np
import cv2
import os

# Carica il modello
model_pose = YOLO("models/pose/weights/best.pt")

# Cartelle dei video
video_dirs = [
    '../video-dataset/violent/cam1/',
    '../video-dataset/violent/cam2/',
    ##'../video-dataset/non-violent/cam1/',
    ##'../video-dataset/non-violent/cam2/'
]

# INDICI DEI KEYPOINT NEL FORMATO COCO (17 KEYPOINTS)
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

def main():

    for path_to_video in video_dirs:
        # Imposta la label in base alla cartella
        if 'non-violent' in path_to_video:
            default_label = 0
        else:
            default_label = 1
        
        print(f"{default_label}")

        if 'cam1' in path_to_video:
            cam_label = 'cam1'
        else:
            cam_label = 'cam2'

        for filename in natsorted(os.listdir(path_to_video)):
            if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
                continue  # salta file non video

            print(f"Elaboro video: {filename} in cartella {path_to_video}")

            people_data = {}  # {person_id: [frames]} dati per ogni persona per video

            # Avvia il tracking sul video 
            results_generator = model_pose.track(
                path_to_video + filename,
                tracker="botsort.yaml",
                #persist=True,
                stream=True,
                verbose=False
            )
            
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

                if default_label == 1:
                    video_display(frame_result)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            cv2.destroyAllWindows()

            if default_label == 1:
                violent_ids = input(f"Inserisci gli id violenti separati da spazio {list(people_data.keys())}: ").split()
                violent_ids = [int(x) for x in violent_ids]

                for pid, frames in people_data.items():
                    if len(frames) == 0:
                        continue  # salta persone senza frame

                    # assegna label 1 se è stato selezionato come violento, altrimenti 0
                    label = 1 if pid in violent_ids else 0

                    # salva i keypoint di questa persona
                    temp_dict = {pid: frames}
                    save_keypoints_to_dataset(temp_dict, filename, label, cam_label, default_label)
            else:
                # Salva i keypoint di tutte le persone con label 0
                save_keypoints_to_dataset(people_data, filename, default_label, cam_label)

            print(f"People data collected so far: {len(people_data)} individuals.")
        

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
def save_keypoints_to_dataset(people_data, filename, label, cam_label, default_label = 0):
    save_path='models/violentvideo/'
    os.makedirs(save_path, exist_ok=True)

    for pid, frames in people_data.items():
        if len(frames) == 0:
            continue  # salta persone senza frame

        person_array = np.array(frames, dtype=np.float32)  # (num_frames, 17, 3)
        video_name = os.path.splitext(os.path.basename(filename))[0]

        if default_label == 1:
            np.savez(f"{save_path}violent_video{video_name}_{cam_label}_person{pid}", data=person_array, label=label)
            print(f"Salvo video violento: {video_name}, {cam_label}, persona ID: {pid}, frames: {person_array.shape[0]}")
        else:
            np.savez(f"{save_path}nonviolent_video{video_name}_{cam_label}_person{pid}", data=person_array, label=label)
            print(f"Salvo video non-violento: {video_name}, {cam_label}, persona ID: {pid}, frames: {person_array.shape[0]}")

# Funzione per visualizzare il video con i box e i keypoint annotati
def video_display(frame_result):
    annotated_frame = frame_result.plot()
    display_frame = cv2.resize(
        annotated_frame, (1080, 720), interpolation=cv2.INTER_AREA)
    # Mostra il frame nella finestra ridimensionabile
    cv2.imshow("Video Annotation", display_frame)


if __name__ == '__main__':
    main()
