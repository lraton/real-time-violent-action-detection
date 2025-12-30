from multiprocessing import Pool, cpu_count
from ultralytics import YOLO
from natsort import natsorted
import numpy as np
import torch
import cv2
import os

# Carica il modello
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_pose = YOLO("../models/yolo11n-pose.pt")
model_pose.to(device)

# Cartelle dei video
video_dirs = [
    '../video-dataset/violent/cam1/',
    '../video-dataset/violent/cam2/',
    '../video-dataset/violent/cam3/',
    '../video-dataset/violent/cam4/',
    '../video-dataset/non-violent/cam1/',
    '../video-dataset/non-violent/cam2/',
    '../video-dataset/non-violent/cam3/',
    '../video-dataset/non-violent/cam4/',
    '../video-dataset/non-violent/cam5/'
]

# INDICI DEI KEYPOINT NEL FORMATO COCO (17 KEYPOINTS)
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6


def process_video_folder(path_to_video):
    # Imposta la label in base alla cartella
    if 'non-violent' in path_to_video:
        default_label = 0
    else:
        default_label = 1

    print(f"{default_label}")

    if 'cam1' in path_to_video:
        cam_label = 'cam1'
    elif 'cam2' in path_to_video:
        cam_label = 'cam2'
    elif 'cam3' in path_to_video:
        cam_label = 'cam3'
    elif 'cam4' in path_to_video:
        cam_label = 'cam4'
    else:
        cam_label = 'cam5'

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
            verbose=False)

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

                    if person_id not in people_data:  # Inizializza la lista per la persona se non esiste
                        people_data[person_id] = []
                    people_data[person_id].append(relative_keypoints_with_conf)  # Aggiungi i keypoint relativi alla lista della persona
            '''
                if default_label == 1:
                    video_display(frame_result)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                '''
        cv2.destroyAllWindows()
        if default_label == 1:
            # --- VIDEO VIOLENTI (Filtraggio Intelligente) ---
            print(f"Analisi video violento: {filename}...")

            violent_people = {}  # Chi combatte (Label 1)
            static_people = {}   # Chi guarda (Label 0 - Ottimo per ridurre falsi positivi!)

            for pid, frames in people_data.items():
                # Filtro Lunghezza (butta via rumore)
                if len(frames) < 30:
                    continue

                # Calcolo Score Movimento
                score = calculate_movement_score(frames)

                # Uno spettatore o un combattente?
                if score < 1.5:
                    print(f"  -> ID {pid} classificato STATICO (score {score:.4f})")
                    static_people[pid] = frames
                else:
                    print(f"  -> ID {pid} classificato VIOLENTO (score {score:.4f})")
                    violent_people[pid] = frames

            # --- SALVATAGGIO DEI DUE GRUPPI ---
            
            # Salva i VIOLENTI (Label 1)
            if len(violent_people) > 0:
                save_keypoints_to_dataset(violent_people, filename, cam_label, default_label)

            # Salva gli SPETTATORI (Label 0)
            if len(static_people) > 0:
                save_keypoints_to_dataset(static_people, filename, cam_label, 0)

        else:
            # --- VIDEO NON VIOLENTI  ---
            cleaned_people_data = {pid: fr for pid, fr in people_data.items() if len(fr) > 15}
            if len(cleaned_people_data) > 0:
                save_keypoints_to_dataset(cleaned_people_data, filename, cam_label, default_label)

        print(f"People data collected so far: {len(people_data)} individuals.")


def calculate_movement_score(frames):
    # Converti in numpy array se non lo è
    data = np.array(frames)

    # Prendiamo solo le prime 2 colonne (X, Y relativi)
    # Assumiamo che la forma sia (frames, 17, 3) dove 3 è (x, y, conf)
    coords = data[:, :, :2]

    # Calcola la differenza (velocità) tra frame consecutivi
    # diffs sarà (N-1, 17, 2)
    diffs = np.diff(coords, axis=0)

    # Calcola la magnitudine del movimento per ogni keypoint (distanza euclidea)
    # magnitudes sarà (N-1, 17)
    magnitudes = np.linalg.norm(diffs, axis=2)

    # Somma il movimento di tutti i keypoint per ogni frame, poi fai la media temporale
    # Questo ci dà un numero unico: "quanto si muove mediamente questo scheletro per frame"
    avg_movement = np.mean(np.sum(magnitudes, axis=1))

    return avg_movement


# Funzione per normalizzare i keypoint rispetto al centro del torace
def normalize_keypoints_relative_to_torso(i, keypoints_normalized):
    person_keypoints = keypoints_normalized[i] # Shape (17, 2)

    left_shoulder = person_keypoints[LEFT_SHOULDER_IDX]
    right_shoulder = person_keypoints[RIGHT_SHOULDER_IDX]

    # Verifica se le spalle sono state rilevate (conf > 0 o coord != 0)
    # YOLO normalizzato a volte mette 0,0 quando non rileva
    if left_shoulder[0] != 0 and right_shoulder[0] != 0:
        
        # Trova il centro del torso
        center_point = (left_shoulder + right_shoulder) / 2

        # Calcola la larghezza (Fattore di scala)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Protezione matematica: se per assurdo la larghezza è 0 (punti sovrapposti)
        if shoulder_width < 0.01: 
            shoulder_width = 1.0 # Evita esplosione dei numeri

        # Normalizzazione completa (Centratura + Scala)
        relative_keypoints = (person_keypoints - center_point) / shoulder_width

    else:
        # FALLBACK: Se non vede le spalle, i dati sono probabilmente sporchi.
        center_point = np.array([0.5, 0.5]) 
        relative_keypoints = person_keypoints - center_point

    return relative_keypoints


#   Salva i keypoint normalizzati nel dataset
def save_keypoints_to_dataset(people_data, filename, cam_label, default_label=0):
    save_path = '../datasets/lstm_dataset/'
    os.makedirs(save_path, exist_ok=True)

    for pid, frames in people_data.items():
        if len(frames) == 0:
            continue  # salta persone senza frame

        person_array = np.array(frames, dtype=np.float32)  # (num_frames, 17, 3)
        video_name = os.path.splitext(os.path.basename(filename))[0]

        if default_label == 1:
            np.savez(f"{save_path}violent_video{video_name}_{cam_label}_person{pid}", data=person_array, label=default_label)
            print(f"Salvo video violento: {video_name}, {cam_label}, persona ID: {pid}, frames: {person_array.shape[0]}")
        else:
            np.savez(f"{save_path}nonviolent_video{video_name}_{cam_label}_person{pid}", data=person_array, label=default_label)
            print(f"Salvo video non-violento: {video_name}, {cam_label}, persona ID: {pid}, frames: {person_array.shape[0]}")


# Funzione per visualizzare il video con i box e i keypoint annotati
def video_display(frame_result):
    annotated_frame = frame_result.plot()
    display_frame = cv2.resize(annotated_frame, (1080, 720), interpolation=cv2.INTER_AREA)
    # Mostra il frame nella finestra ridimensionabile
    cv2.imshow("Video Annotation", display_frame)


def main():
    # Usa metà dei core disponibili per sicurezza
    num_workers = min(len(video_dirs), max(1, cpu_count() // 2))
    print(f"Avvio {num_workers} processi in parallelo...")

    # Esegui il processing in parallelo
    with Pool(num_workers) as pool:
        pool.map(process_video_folder, video_dirs)


if __name__ == '__main__':
    main()
