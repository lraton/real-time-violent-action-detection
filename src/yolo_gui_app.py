import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
from ultralytics import YOLO

# --- Funzione colori ---
def get_colours(cls_num: int) -> tuple[int, int, int]:
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# --- Skeleton della pose ---
SKELETON = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (11, 12),
    (5, 11), (6, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

class YOLOCameraApp:
    def __init__(self, knife_model_path="yolo11n.pt", pose_model_path="yolo11n-pose.pt"):
        self.model_knife = YOLO(knife_model_path)
        self.model_pose = YOLO(pose_model_path)

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
        ttk.Label(self.frame_right, text="Oggetti rilevati:", font=("Arial", 14)).pack()
        self.object_list = tk.Listbox(self.frame_right, width=30, font=("Arial", 12))
        self.object_list.pack(fill=tk.Y, expand=True)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

    # --- Detection oggetti ---
    def detect_objects(self, frame):
        detected = set()
        results_det = self.model_knife(frame, verbose=False)
        for result in results_det:
            class_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = class_names[cls]
                    conf = float(box.conf[0])
                    detected.add(f"{class_name} {conf:.2f}")

                    colour = get_colours(cls)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, colour, 2)
        return detected

    # --- Pose ---
    def detect_pose(self, frame):
        detected_persons = set()
        results_pose = self.model_pose(frame, verbose=False)
        for result in results_pose:
            if result.keypoints is not None:
                kpts_list = result.keypoints.xy
                for idx, person in enumerate(kpts_list):
                    colour = get_colours(idx + 100)
                    # Disegna punti
                    for x, y in person:
                        cv2.circle(frame, (int(x), int(y)), 4, colour, -1)
                    # Scheletro
                    for (i, j) in SKELETON:
                        x1, y1 = person[i]
                        x2, y2 = person[j]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
                    # Box attorno alla persona (se disponibile)
                    if result.boxes is not None and len(result.boxes.xyxy) > idx:
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(frame, f"Persona {idx}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                    detected_persons.add(f"Persona {idx}")
        return frame, detected_persons


    # --- Aggiornamento frame ---
    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.root.after(10, self.update_frame)
            return

        # Detection oggetti
        detected_items = self.detect_objects(frame)

        # Pose + box persone
        frame, detected_persons = self.detect_pose(frame)

        # Unisci oggetti e persone
        all_detected = detected_items.union(detected_persons)

        # Aggiorna lista a destra
        self.object_list.delete(0, tk.END)
        for obj in all_detected:
            self.object_list.insert(tk.END, obj)

        # Aggiorna immagine webcam
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


# --- Avvio ---
if __name__ == "__main__":
    app = YOLOCameraApp()
    app.run()
