import cv2
import os
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
from ultralytics import YOLO

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
    def __init__(self, knife_model_path="models/knife/runs/detect/train3/weights/best.pt", pose_model_path="yolo11n-pose.pt"):
        self.model_knife = YOLO(knife_model_path)
        self.model_pose = YOLO(pose_model_path)
        
        self.saved_faces_ids = set()  # Per evitare salvataggi duplicati di volti

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
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

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
        results_det = self.model_knife(frame, verbose=False)

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
        return detected  # Restituisce solo i dati, non il frame modificato

    # --- Pose ---

    def detect_pose(self, frame, detected_items):
        for obj in detected_items:
            x1, y1, x2, y2 = obj["box"]
            class_name = obj["class"]
            conf = obj["conf"]
            cls = obj["cls"]

            colour = get_colours(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2)

        clean_frame_for_save = frame.copy()

        detected_persons = set()
        results_pose = self.model_pose.track(
            frame, tracker="botsort.yaml", persist=True, verbose=False)

        for result in results_pose:
            if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
                kpts_list = result.keypoints.xy
                ids = result.boxes.id.cpu().numpy().astype(int).tolist()

                # Sicurezza: a volte il numero di ID non corrisponde al numero di pose rilevate
                if len(ids) != len(kpts_list):
                    ids = [idx for idx in range(len(kpts_list))]

                for idx, person in enumerate(kpts_list):
                    person_id = ids[idx]

                    # Bounding box della persona
                    if result.boxes is not None and len(result.boxes.xyxy) > idx:
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx])
                        person_box = (x1, y1, x2, y2)

                        inside = False
                        for obj in detected_items:
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

                        # Colore per la visualizzazione sullo schermo
                        colour = get_colours(person_id + 100)
                        box_color = (0, 0, 255) if inside else colour

                        if inside:
                            # Salva il volto sospetto solo se non è già stato salvato
                            if person_id not in self.saved_faces_ids:
                                # Estrai e salva il volto sospetto
                                self.extract_suspicious_face(
                                    clean_frame_for_save, person, person_box, person_id)
                            else:
                                print(
                                    f"Volto sospetto per ID {person_id} già salvato.")

                            cv2.putText(frame, f"Sospetto {person_id}", (x1, max(
                                y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                            detected_persons.add(
                                f"Sospetto {person_id} (ARMA)")
                        else:
                            cv2.putText(frame, f"Persona {person_id}", (x1, max(
                                y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                            detected_persons.add(f"Persona {person_id}")

                        # Disegna la box della persona sul frame per la visualizzazione
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # Disegna punti e scheletro (Parte finale del disegno)
                        for x, y in person:
                            cv2.circle(frame, (int(x), int(y)), 4, colour, -1)
                        for (i, j) in SKELETON:
                            x1_k, y1_k = person[i]
                            x2_k, y2_k = person[j]
                            cv2.line(frame, (int(x1_k), int(y1_k)),
                                     (int(x2_k), int(y2_k)), colour, 2)

        return frame, detected_persons

    # --- Aggiornamento frame ---
    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.root.after(10, self.update_frame)
            return

        detected_items = self.detect_objects(frame)

        # Pose: Ritorna il frame con i disegni
        frame_drawn, detected_persons = self.detect_pose(
            frame, detected_items)  # Passa il frame originale

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
        self.root.after(10, self.update_frame)

    # --- Avvio app ---
    def run(self):
        self.update_frame()
        self.root.mainloop()
        self.cap.release()
