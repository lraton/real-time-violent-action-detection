import cv2
import os
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np

camera_index = 1  # Modifica questo indice se necessario

# --- Funzione colori ---
def get_colours(cls_num: int) -> tuple[int, int, int]:
    random.seed(cls_num)
    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # Reject strong red tones
        if not (r > 180 and g < 100 and b < 100):
            return (r, g, b)


# --- Skeleton della pose ---
SKELETON = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (11, 12),
    (5, 11), (6, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]


class YOLOCameraApp:

    model = load_model("models/lstm_violence_detector.keras")

    # Costanti per rilevamento violenza
    VIOLENCE_THRESHOLD = 0.7         # Soglia per classificare come violento
    COLOR_VIOLENT = (0, 0, 255)      # Rosso (BGR) per comportamento violento
    COLOR_NON_VIOLENT = (0, 255, 0)  # Verde (BGR) per comportamento non-violento
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    PERSON_PREFIX = "Persona"
    STATUS_TEXT = "Non-Violenta"

    def __init__(self, knife_model_path="models/knife/weights/best.pt", pose_model_path="models/yolo11n-pose.pt"):
        self.model_knife = YOLO(knife_model_path)
        self.model_pose = YOLO(pose_model_path)
        
        self.saved_faces_ids = set()  # Per evitare salvataggi duplicati di volti
        self.person_sequences = {} # Sequenze di keypoints per ogni persona ID
        # Dizionario per l'ultima predizione di violenza per ogni ID
        self.person_last_prediction = {} 
        # Dizionario per il contatore di frame per ogni ID
        self.person_frame_counter = {}   
        # Esegui l'LSTM solo 1 volta ogni N frame (prova con 3 o 5)
        self.PREDICTION_SKIP_FRAMES = 3

        self.prev_time = 0  # Tempo del frame precedente
        self.fps = 0        # Valore FPS calcolato

        # Tkinter GUI
        self.root = tk.Tk()
        self.root.title("YOLO Pose + Object Detection GUI")

        # Frame webcam
        self.frame_left = tk.Frame(self.root)
        self.frame_left.pack(side=tk.LEFT)
        self.lmain = tk.Label(self.frame_left)
        self.lmain.pack()

        # Frame oggetti rilevati
        self.frame_right = tk.Frame(self.root)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(self.frame_right, text="Oggetti rilevati:",
                  font=("Arial", 14)).pack()
        self.object_list = tk.Listbox(
            self.frame_right, width=30, font=("Arial", 12))
        self.object_list.pack(fill=tk.Y, expand=True)

        # Webcam
        self.cap = cv2.VideoCapture("../video-dataset/non-violent/cam1/1.mp4")
        self.cap.set(3, 640)
        self.cap.set(4, 480)

    # --- Estrazione Volto Sospetto ---
    def extract_suspicious_face(self, frame, person_keypoints, person_box, person_id):
        # person_keypoints è un array di 17 keypoints (x, y)
        # Keypoints: Naso (0), Occhi (1, 2), Orecchie (3, 4)
        face_kpts = person_keypoints[:5]

        # Se non ci sono keypoints per la faccia o sono a (0, 0), usciamo
        if len(face_kpts) == 0 or face_kpts.min() <= 0:
            return None

        # 1. Calcola il bounding box stretto dai keypoints della faccia
        min_x = int(face_kpts[:, 0].min())
        max_x = int(face_kpts[:, 0].max())
        min_y = int(face_kpts[:, 1].min())
        max_y = int(face_kpts[:, 1].max())

        # Calcola l'altezza e la larghezza del box iniziale
        kpt_height = max_y - min_y
        kpt_width = max_x - min_x

        # 2. Estendiamo in larghezza per catturare l'intera testa (che è più larga della distanza tra le orecchie)
        width_expansion = int(kpt_width * 0.1)

        # Estendiamo in altezza in alto e in basso
        y_expansion_top = int(kpt_height * 2.5)
        # Espansione in basso (per mento e collo)
        y_expansion_bottom = int(kpt_height * 1)

        # 3. Calcola il nuovo Bounding Box
        x1_new = max(0, min_x - width_expansion)
        y1_new = max(0, min_y - y_expansion_top)
        x2_new = min(frame.shape[1], max_x + width_expansion)
        y2_new = min(frame.shape[0], max_y + y_expansion_bottom)

        x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new

        # 4. Se il box è troppo piccolo, usa il box della persona per stimare la faccia
        if x2 - x1 < 80 or y2 - y1 < 80:  # Aumentato il requisito di dimensione minima a 80x80
            p_x1, p_y1, p_x2, p_y2 = person_box
            x1 = p_x1
            y1 = p_y1
            x2 = p_x2
            y2 = p_y1 + (p_y2 - p_y1) // 3

        # 5. Taglia e restituisce il volto
        os.makedirs("../suspect/", exist_ok=True)
        face_image = frame[y1:y2, x1:x2]
        if face_image is not None and face_image.size > 0:
            face_filename = f"../suspect/face_susp_{person_id}_"+time.strftime(
                '%b-%d-%Y_%H%M')+".jpg"
            cv2.imwrite(face_filename, face_image)
            print(
                f"Volto sospetto salvato: {face_filename}")
            self.saved_faces_ids.add(person_id)

    # --- Detection oggetti ---
    def detect_objects(self, frame):
        detected = []
        # Rimuovi i risultati per poter usare solo i detection per il disegno.
        results_det = self.model_knife(
            frame, 
            verbose=False, 
            imgsz=320,  # Ridimensiona l'input del modello
            half=True,  # Usa precisione FP16 (solo GPU)
            device=0,    # Forza l'uso della GPU (es. 'cuda' o 0)
            conf=0.55
        )

        # Crea una lista di oggetti rilevati con tutte le info necessarie
        for result in results_det:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = class_names[cls]
                    conf = float(box.conf[0])

                    detected.append({
                        "class": class_name,
                        "conf": conf,
                        "box": (x1, y1, x2, y2),
                        "cls": cls  # Aggiungiamo cls per i colori nel disegno successivo
                    })
                    
                    colour = get_colours(cls)
                    # Disegna la box dell'oggetto sul frame per la visualizzazione
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)
        return detected  # Restituisce solo i dati, non il frame modificato

    # --- Pose ---
    def detect_pose(self, frame, detected_items):
        # Disegna gli oggetti rilevati prima
        for obj in detected_items:
            x1, y1, x2, y2 = obj["box"]

        clean_frame_for_save = frame.copy()

        detected_persons = set()
        results_pose = self.model_pose.track(
                    frame, 
                    tracker="botsort.yaml", 
                    persist=True, 
                    verbose=False,
                    imgsz=320,  # Ridimensiona l'input del modello
                    half=True,  # Usa precisione FP16 (solo GPU)
                    device=0,    # Forza l'uso della GPU (es. 'cuda' o 0)
                    conf=0.6
                )
        
        # Otteniamo gli ID attualmente tracciati nel frame
        current_ids_in_frame = set()
        if results_pose:
            for result in results_pose:
                if result.boxes is not None and result.boxes.id is not None:
                    current_ids_in_frame.update(result.boxes.id.cpu().numpy().astype(int).tolist())

        # Rimuovi le sequenze delle persone non più tracciate
        ids_to_remove = set(self.person_sequences.keys()) - current_ids_in_frame
        for old_id in ids_to_remove:
            # print(f"Rimuovo la sequenza per l'ID {old_id} non più tracciato.")
            del self.person_sequences[old_id]
        
        # Rimuovi le predizioni delle persone non più tracciate
        ids_to_remove_pred = set(self.person_last_prediction.keys()) - current_ids_in_frame
        for old_id in ids_to_remove_pred:
            if old_id in self.person_last_prediction:
                del self.person_last_prediction[old_id]
            if old_id in self.person_frame_counter:
                del self.person_frame_counter[old_id]

        for result in results_pose:
            frame=result.plot()  # Disegna i keypoints sul frame
            if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
                kpts_list = result.keypoints.xy
                kpts_normalized = result.keypoints.xyn.cpu().numpy() # PER LA PREDIZIONE (come nel training)
                ids = result.boxes.id.cpu().numpy().astype(int).tolist() # IDs delle persone
                kpts_conf = result.keypoints.conf.cpu().numpy() # confidence dei keypoints

                # Sicurezza: a volte il numero di ID non corrisponde al numero di pose rilevate
                if len(ids) != len(kpts_list):
                    ids = [idx for idx in range(len(kpts_list))]

                for idx, person in enumerate(kpts_list):

                    # Prendiamo i dati specifici per questa persona
                    person_kpts_normalized = kpts_normalized[idx]
                    person_id = ids[idx]
                    person_kpts_conf = kpts_conf[idx] # confidence dei keypoints

                    # Bounding box della persona
                    if result.boxes is not None and len(result.boxes.xyxy) > idx:
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx])
                        person_box = (x1, y1, x2, y2)

                        inside = False 
                        for obj in detected_items: # Controlla se la persona è dentro un oggetto rilevato (es. coltello)
                            ox1, oy1, ox2, oy2 = obj["box"]
                            # Calcola le coordinate dell'intersezione
                            overlap_x1 = max(x1, ox1)
                            overlap_y1 = max(y1, oy1)
                            overlap_x2 = min(x2, ox2)
                            overlap_y2 = min(y2, oy2)

                            # Calcola larghezza e altezza dell'area di sovrapposizione
                            overlap_w = max(0, overlap_x2 - overlap_x1)
                            overlap_h = max(0, overlap_y2 - overlap_y1)

                            # Calcola l'area di intersezione
                            intersection_area = overlap_w * overlap_h
                            if intersection_area > 0:
                                inside = True
                                
                                break

                        # Predizione della violenza basata sui keypoints della persona
                        violence_score = self.predict_violence(person_kpts_normalized, person_kpts_conf, person_id)

                        box_color = self.COLOR_NON_VIOLENT  # Default non-violento
                        is_violent = False
                        detection_entry = ""
                        # Se la predizione ha restituito un punteggio valido
                        if violence_score is not None:
                            is_violent = violence_score > self.VIOLENCE_THRESHOLD
                            
                            if is_violent:
                                self.STATUS_TEXT = "VIOLENTA"
                                box_color = self.COLOR_VIOLENT
                            else:
                                self.STATUS_TEXT = "Non-violenta"
                                box_color = self.COLOR_NON_VIOLENT

                        # Logica per i "sospetti" (persone con coltello)
                        if inside:
                            self.PERSON_PREFIX = "Sospetto"
                            box_color = self.COLOR_VIOLENT
                        else:
                            self.PERSON_PREFIX = "Persona"
                            box_color = self.COLOR_NON_VIOLENT
                        
                        if is_violent or person_id not in self.saved_faces_ids:
                                # Salva il volto del sospetto, ma solo la prima volta che viene rilevato
                                self.extract_suspicious_face(
                                    clean_frame_for_save, person, person_box, person_id
                                )
                            
                        if violence_score is None:
                            detection_entry = f"{self.PERSON_PREFIX} {person_id} | {self.STATUS_TEXT} (N/A)"
                        else:
                            detection_entry = f"{self.PERSON_PREFIX} {person_id} | {self.STATUS_TEXT} ({violence_score:.2f})"                     
                        # Aggiungi alla lista degli oggetti rilevati
                        detected_persons.add(detection_entry)

                        

                        # Etichetta finale da visualizzare
                        if violence_score is None:
                            final_label = f"{self.PERSON_PREFIX} {person_id} | {self.STATUS_TEXT} (N/A)"
                        else:
                            final_label = f"{self.PERSON_PREFIX} {person_id} | {self.STATUS_TEXT} ({violence_score:.2f})"
                        text_position = (x1, max(y1 - 10, 20))
                        
                        cv2.putText(frame, final_label, text_position, self.FONT, self.FONT_SCALE, box_color, self.FONT_THICKNESS)
                        '''
                        # Disegna il bounding box della persona
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # Disegna punti e scheletro (Parte finale del disegno)
                        for x, y in person:
                            cv2.circle(frame, (int(x), int(y)), 4, box_color, -1)
                        for (i, j) in SKELETON:
                            x1_k, y1_k = person[i]
                            x2_k, y2_k = person[j]
                            cv2.line(frame, (int(x1_k), int(y1_k)),
                                     (int(x2_k), int(y2_k)), box_color, 2)'''            
                        
        return frame, detected_persons
    
    # Normalizza i keypoints rispetto al torso
    def normalize_keypoints_relative_to_torso(self, person_keypoints_xyn):

        left_shoulder = person_keypoints_xyn[5]
        right_shoulder = person_keypoints_xyn[6]

        if left_shoulder.sum() > 0 and right_shoulder.sum() > 0:
            center_point = (left_shoulder + right_shoulder) / 2
        else:
            center_point = np.array([0.0, 0.0])

        relative_keypoints = person_keypoints_xyn - center_point

        return relative_keypoints
    
    # Predizione violenza
    def predict_violence(self, person_keypoints_normalized, person_kpts_conf, person_id):
        print(f"Predicting violence for person ID: {person_id} frame : {len(self.person_sequences[person_id]) if person_id in self.person_sequences else 'new person'}")

        relative_kpts_xy = self.normalize_keypoints_relative_to_torso(person_keypoints_normalized)
            
        keypoints_with_conf = np.hstack([relative_kpts_xy, person_kpts_conf[:, None]])  # Aggiungi la conf come terza colonna

        flattened = keypoints_with_conf.flatten()

        # Ottieni o crea la sequenza specifica per QUESTA persona
        if person_id not in self.person_sequences:
            self.person_sequences[person_id] = deque(maxlen=150)
            # Inizia da 0 per predire subito al primo frame utile
            self.person_frame_counter[person_id] = 0 
            # Nessuna predizione ancora
            self.person_last_prediction[person_id] = None 
        
        current_sequence = self.person_sequences[person_id]
        current_sequence.append(flattened)

        # Se la sequenza non è piena, non possiamo predire nulla
        if len(current_sequence) < 150:
            return None 
        
        # Controlla se è il momento di predire (contatore a 0)
        if self.person_frame_counter[person_id] > 0:
            # Se non è il momento, decrementa il contatore
            self.person_frame_counter[person_id] -= 1
            # E restituisci l'ULTIMA predizione nota
            return self.person_last_prediction[person_id] 
        
        # Se il contatore è a 0, ESEGUI la predizione
        # e resetta il contatore
        self.person_frame_counter[person_id] = self.PREDICTION_SKIP_FRAMES 

        data = np.array(current_sequence, dtype=np.float32).reshape(1, 150, -1)
        pred = self.model.predict(data, verbose=0)[0][0]
        
        # Salva questa nuova predizione come "ultima predizione nota"
        self.person_last_prediction[person_id] = float(pred) 
        
        return float(pred)


    # --- Aggiornamento frame ---
    def update_frame(self):
        current_time = time.time()
        if self.prev_time > 0:
            elapsed_time = current_time - self.prev_time
            # Evita divisione per zero se il tempo è troppo piccolo
            if elapsed_time > 0:
                self.fps = 1.0 / elapsed_time
        self.prev_time = current_time

        success, frame = self.cap.read()
        if not success:
            self.root.after(10, self.update_frame)
            return

        detected_items = self.detect_objects(frame)

        # Pose: Ritorna il frame con i disegni
        frame_drawn, detected_persons = self.detect_pose(
            frame, detected_items)  # Passa il frame originale
        
        fps_text = f"FPS: {self.fps:.1f}"
        # Posizionalo in alto a sinistra (coordinate 10, 30)
        cv2.putText(frame_drawn, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA) # Aggiungi il testo FPS al frame

        # Unisci oggetti e persone
        object_names = {
            f"{obj['class']} {obj['conf']:.2f}" for obj in detected_items}
        all_detected = object_names.union(detected_persons)

        # Aggiorna lista a destra
        self.object_list.delete(0, tk.END)
        for obj in all_detected:
            self.object_list.insert(tk.END, obj)

        # Aggiorna immagine webcam
        cv2image = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

        # Richiamo ricorsivo
        self.root.after(50, self.update_frame)

    # --- Avvio app ---
    def run(self):
        self.update_frame()
        self.root.mainloop()
        self.cap.release()
