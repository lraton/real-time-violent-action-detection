import cv2
import os
import time
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from keras.models import load_model


class ViolenceDetectionSystem:

    # Costanti per la classe
    VIOLENCE_THRESHOLD = 0.7
    COLOR_VIOLENT = (0, 0, 255)  # Rosso (BGR)
    COLOR_NON_VIOLENT = (0, 255, 0)  # Verde (BGR)
    COLOR_KNIFE = (255, 0, 0)  # Blu (BGR)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    # Scheletro
    SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    def __init__(self, knife_model_path, pose_model_path):
        print("Caricamento modelli in corso...")
        # Carica il modello LSTM per la rilevazione della violenza
        self.model_lstm = load_model("../models/lstm_violence_detector.keras")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_knife = YOLO(knife_model_path)
        self.model_pose = YOLO(pose_model_path)
        self.model_pose.to(device)
        self.model_knife.to(device)
        print("Modelli caricati.")

        self.saved_faces_ids = set()  # Per evitare salvataggi duplicati
        self.person_sequences = {}  # Sequenze di keypoints per ogni persona ID

    # --- METODO PRINCIPALE CHIAMATO DAL MAIN ---
    def process_frame(self, frame):
        t_start = time.time()
        # Salva una copia pulita per l'estrazione dei volti
        clean_frame_for_save = frame.copy()

        # Rileva oggetti (coltelli)
        t_obj_start = time.time()
        detected_items = self.detect_objects(frame)
        t_obj = (time.time() - t_obj_start) * 1000  # in ms

        # Prende le pose e traccia le persone
        t_pose_start = time.time()
        results_pose = self.model_pose.track(frame, tracker="botsort.yaml", persist=True, verbose=False, imgsz=320, half=True, conf=0.6)
        t_pose = (time.time() - t_pose_start) * 1000  # in ms

        # Processa i risultati delle pose
        t_logic_start = time.time()
        person_data, person_strings_for_list = self.detect_pose(results_pose, detected_items, clean_frame_for_save)
        t_logic = (time.time() - t_logic_start) * 1000  # in ms

        # Disegna box persone
        t_draw_start = time.time()
        if results_pose and results_pose[0]:
            frame_drawn = frame.copy()
            result = results_pose[0]
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
            for data in person_data:
                cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), data["color"], 2)
        else:
            frame_drawn = frame
        t_draw = (time.time() - t_draw_start) * 1000  # in ms

        # Disegna oggetti e etichette persone
        frame_drawn = self.draw_detections(frame_drawn, detected_items, person_data)

        # Prepara la lista di stringhe per la GUI
        object_strings = {f"{obj['class']} {obj['conf']:.2f}" for obj in detected_items}
        all_detected_strings = object_strings.union(person_strings_for_list)

        t_total = (time.time() - t_start) * 1000

        # --- Stampa i risultati nel terminale ---
        
        print(f"--- ANALISI FRAME (ms) ---")
        print(f"  1. Knife Detect : {t_obj:.1f} ms")
        print(f"  2. Pose Detect  : {t_pose:.1f} ms")
        print(f"  3. Logic/LSTM   : {t_logic:.1f} ms")
        print(f"  4. Drawing      : {t_draw:.1f} ms")
        print(f"  --------------------------")
        print(f"  TOTALE FRAME  : {t_total:.1f} ms  (Target: {1000/t_total:.1f} FPS)")
        print("\n")  # Aggiungi uno spazio
        
        return frame_drawn, all_detected_strings

    # --- RILEVAMENTO OGGETTI ---
    def detect_objects(self, frame):
        detected = []

        results_det = self.model_knife(frame, verbose=False, imgsz=320, half=True, conf=0.6)  # Rileva oggetti (coltelli)

        for result in results_det:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    class_name = class_names[cls_id]
                    conf = float(box.conf[0])
                    detected.append({"class": class_name, "conf": conf, "box": (x1, y1, x2, y2), "cls_id": cls_id})
        return detected

    # --- RILEVAMENTO POSE E ANALISI VIOLENZA ---
    def detect_pose(self, results_pose, detected_items, clean_frame):

        person_data_for_drawing = []  # Lista di dizionari per 'draw_detections'
        person_strings_for_list = set()  # Set di stringhe per la lista GUI

        # Itera sui risultati
        for result in results_pose:
            if result.keypoints is None or result.boxes is None or result.boxes.id is None:
                continue

            kpts_list_xy = result.keypoints.xy.cpu().numpy()
            kpts_normalized_xyn = result.keypoints.xyn.cpu().numpy()
            kpts_conf = result.keypoints.conf.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int).tolist()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int).tolist()

            for idx, person_id in enumerate(ids):  # Itera sulle persone tracciate
                if idx >= len(kpts_list_xy): continue  # Sicurezza

                person_kpts_xy = kpts_list_xy[idx]
                person_kpts_xyn = kpts_normalized_xyn[idx]
                person_kpts_conf = kpts_conf[idx]
                person_box = boxes_xyxy[idx]
                x1, y1, x2, y2 = person_box

                # Controlla se la persona ha un oggetto (es. coltello)
                is_suspect = False
                for obj in detected_items:
                    ox1, oy1, ox2, oy2 = obj["box"]
                    # Semplice controllo di sovrapposizione (Intersection over Union > 0)
                    overlap_x1 = max(x1, ox1)
                    overlap_y1 = max(y1, oy1)
                    overlap_x2 = min(x2, ox2)
                    overlap_y2 = min(y2, oy2)
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        is_suspect = True
                        break

                # Predizione della violenza
                violence_score = self.predict_violence(person_kpts_xyn, person_kpts_conf, person_id)

                is_violent = False
                status_text = "Non-Violenta"
                box_color = self.COLOR_NON_VIOLENT
                score_text = "(N/A)"

                # Analizza il risultato della predizione
                if violence_score is not None:
                    is_violent = violence_score > self.VIOLENCE_THRESHOLD
                    score_text = f"({violence_score:.2f})"
                    if is_violent:
                        status_text = "VIOLENTA"
                        box_color = self.COLOR_VIOLENT

                # Analizza il prefisso della persona
                person_prefix = "Persona"
                if is_suspect:
                    person_prefix = "Sospetto CON COLTELLO"
                    box_color = self.COLOR_VIOLENT  # Se è sospetto, colora di rosso

                # Salva il volto se sospetto o violento (e non già salvato)
                if (is_suspect or is_violent) and person_id not in self.saved_faces_ids:
                    self.extract_suspicious_face(clean_frame, person_kpts_xy, person_box, person_id)

                # Prepara i dati
                final_label = f"{person_prefix} {person_id} | {status_text} {score_text} | Confidence: {person_kpts_conf.mean():.2f}"

                text_position = (x1, max(y2 - 10, 20))

                person_strings_for_list.add(final_label)
                person_data_for_drawing.append({"label": final_label, "pos": text_position, "color": box_color})

        return person_data_for_drawing, person_strings_for_list

    # --- DISEGNO --- senza pose (già disegnate da .plot())
    def draw_detections(self, frame, detected_items, person_data):

        # Disegna box oggetti (coltelli)
        for item in detected_items:
            x1, y1, x2, y2 = item["box"]
            label = f"{item['class']} {item['conf']:.2f}"
            color = self.COLOR_KNIFE  # Blu per i coltelli

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), self.FONT, self.FONT_SCALE, color, self.FONT_THICKNESS)

        # Disegna etichette stato persone (Violenta/Sospetto)
        for data in person_data:
            cv2.putText(frame, data["label"], data["pos"], self.FONT, self.FONT_SCALE, data["color"], self.FONT_THICKNESS)

        return frame

    # --- ESTRAZIONE VOLTO ---
    def extract_suspicious_face(self, frame, person_keypoints, person_box, person_id):
        # person_keypoints è (17, 2)
        face_kpts = person_keypoints[:5]  # Naso (0), Occhi (1, 2), Orecchie (3, 4)

        if face_kpts.min() <= 0:  # Se non ci sono keypoints validi
            # Usa un fallback basato sul box della persona
            p_x1, p_y1, p_x2, p_y2 = person_box
            x1, y1 = p_x1, p_y1
            x2, y2 = p_x2, p_y1 + (p_y2 - p_y1) // 3
        else:
            min_x = int(face_kpts[:, 0].min())
            max_x = int(face_kpts[:, 0].max())
            min_y = int(face_kpts[:, 1].min())
            max_y = int(face_kpts[:, 1].max())

            kpt_height = max_y - min_y
            kpt_width = max_x - min_x

            # Evita divisioni per zero se i punti sono coincidenti
            if kpt_height == 0: kpt_height = 20
            if kpt_width == 0: kpt_width = 20

            width_expansion = int(kpt_width * 0.5)  # Espansione laterale
            y_expansion_top = int(kpt_height * 1.5)  # Espansione sopra (capelli)
            y_expansion_bottom = int(kpt_height * 1.0)  # Espansione sotto (mento)

            x1 = max(0, min_x - width_expansion)
            y1 = max(0, min_y - y_expansion_top)
            x2 = min(frame.shape[1], max_x + width_expansion)
            y2 = min(frame.shape[0], max_y + y_expansion_bottom)

        # Taglia e salva
        if x1 < x2 and y1 < y2:
            face_image = frame[y1:y2, x1:x2]
            if face_image.size > 0:
                os.makedirs("../suspect/", exist_ok=True)
                face_filename = f"../suspect/face_susp_{person_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(face_filename, face_image)
                # print(f"Volto sospetto salvato: {face_filename}")
                self.saved_faces_ids.add(person_id)

    #--- NORMALIZZAZIONE KEYPOINT TO TORSO ---
    def normalize_keypoints_relative_to_torso(self, person_keypoints_xyn):
        left_shoulder = person_keypoints_xyn[5]
        right_shoulder = person_keypoints_xyn[6]

        if left_shoulder.sum() > 0 and right_shoulder.sum() > 0:
            center_point = (left_shoulder + right_shoulder) / 2.0
        else:
            # Fallback se le spalle non sono visibili
            center_point = np.array([0.5, 0.5])

        relative_keypoints = person_keypoints_xyn - center_point
        return relative_keypoints

    # --- PREDIZIONE VIOLENZA ---
    def predict_violence(self, person_keypoints_normalized, person_kpts_conf, person_id):

        relative_kpts_xy = self.normalize_keypoints_relative_to_torso(person_keypoints_normalized)
        keypoints_with_conf = np.hstack([relative_kpts_xy, person_kpts_conf[:, None]])
        flattened = keypoints_with_conf.flatten()  # 17 * 3 = 51 features

        if person_id not in self.person_sequences:
            self.person_sequences[person_id] = deque(maxlen=150)

        #print(f"Predicting violence for person ID: {person_id} frame {len(self.person_sequences[person_id])}")

        current_sequence = self.person_sequences[person_id]
        current_sequence.append(flattened)

        if len(current_sequence) < 150:
            return None  # Sequenza non ancora piena

        try:
            data = np.array(current_sequence, dtype=np.float32).reshape(1, 150, 51)
            pred = self.model_lstm.predict(data, verbose=0)[0][0]
            return float(pred)
        except Exception as e:
            print(f"Errore predizione LSTM: {e}")
            return None
