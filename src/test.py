import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
from ultralytics import YOLO

# --- Funzione colori ---
def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

# --- Modelli ---
model_det = YOLO("yolo11n.pt")       # detection
model_pose = YOLO("yolo11n-pose.pt") # pose

skeleton = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (11, 12),
    (5, 11), (6, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

# --- Tkinter GUI ---
root = tk.Tk()
root.title("YOLO Pose + Object Detection GUI")

# Frame sinistro: webcam
frame_left = tk.Frame(root)
frame_left.pack(side=tk.LEFT)

lmain = tk.Label(frame_left)
lmain.pack()

# Frame destro: oggetti rilevati
frame_right = tk.Frame(root)
frame_right.pack(side=tk.RIGHT, fill=tk.Y)

ttk.Label(frame_right, text="Oggetti rilevati:", font=("Arial", 14)).pack()
object_list = tk.Listbox(frame_right, width=30, font=("Arial", 12))
object_list.pack(fill=tk.Y, expand=True)

# Apri webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# --- Funzione aggiornamento ---
def update_frame():
    success, frame = cap.read()
    if not success:
        root.after(10, update_frame)
        return

    detected_objects = set()

    # --- Detection oggetti ---
    results_det = model_det(frame, verbose=False)
    for result in results_det:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = class_names[cls]
                conf = float(box.conf[0])
                detected_objects.add(f"{class_name} {conf:.2f}")
                colour = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

    # --- Pose ---
    results_pose = model_pose(frame, verbose=False)
    if results_pose and results_pose[0].keypoints is not None:
        kpts = results_pose[0].keypoints.xy
        for person in kpts:
            for x, y in person:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            for (i, j) in skeleton:
                x1, y1 = person[i]
                x2, y2 = person[j]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # --- Aggiorna lista oggetti ---
    object_list.delete(0, tk.END)
    for obj in detected_objects:
        object_list.insert(tk.END, obj)

    # --- Aggiorna immagine webcam ---
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # richiamo ricorsivo
    root.after(10, update_frame)

# --- Avvia loop ---
update_frame()
root.mainloop()

cap.release()
