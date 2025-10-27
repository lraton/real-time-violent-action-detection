import cv2
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from violence_detection_system import ViolenceDetectionSystem


def main():
    prev_time = 0  # Tempo del frame precedente
    fps = 0  # Valore FPS calcolato
    camera_index = 1  # Modifica questo indice se necessario (spesso 0 o 1)
    backup_video_path = '../video-dataset/violent/cam1/37.mp4'

    # Inizializza la classe che gestisce YOLO
    app = ViolenceDetectionSystem(knife_model_path="models/knife/weights/best.pt", pose_model_path="models/yolo11n-pose.pt")

    # --- Tkinter GUI ---
    root = tk.Tk()
    root.title("YOLO Pose + Object Detection GUI")

    # Frame webcam
    frame_left = tk.Frame(root)
    frame_left.pack(side=tk.LEFT, padx=10, pady=10)
    lmain = tk.Label(frame_left)
    lmain.pack()

    # Frame oggetti rilevati
    frame_right = tk.Frame(root)
    frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
    ttk.Label(frame_right, text="Persone e Oggetti rilevati:", font=("Arial", 14)).pack()
    object_list = tk.Listbox(frame_right, width=40, font=("Arial", 12))  # Aumentata larghezza
    object_list.pack(fill=tk.Y, expand=True)

    # --- Webcam ---
    print(f"Provo ad aprire la webcam all'indice {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire la webcam all'indice {camera_index}.")
        # Prova ad aprire un video di backup
        print(f"Provo ad aprire il video di backup: {backup_video_path}...")
        cap = cv2.VideoCapture(backup_video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video di backup.")
            root.destroy()
            return

    cap.set(3, 192)
    cap.set(4, 256)

    # Avvia il loop di aggiornamento
    update_frame(root, lmain, cap, object_list, app, prev_time, fps)

    # Avvia la GUI
    root.mainloop()

    # Rilascia la webcam alla chiusura
    cap.release()
    print("Applicazione chiusa, webcam rilasciata.")


# --- Aggiornamento frame ---
def update_frame(root, lmain, cap, object_list, app, prev_time, fps):

    # Calcolo FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 0:
        fps = 1.0 / elapsed_time
    prev_time = current_time  # Salva il tempo corrente per la prossima iterazione

    success, frame = cap.read()
    if not success:
        print("Errore: Impossibile leggere il frame dalla webcam.")
        # Riprova dopo un breve intervallo
        root.after(50, update_frame, root, lmain, cap, object_list, app, prev_time, fps)
        return
    try:
        frame_drawn, all_detected_strings = app.process_frame(frame)
    except Exception as e:
        print(f"Errore durante process_frame: {e}")
        # Continua a mostrare il frame originale in caso di errore
        frame_drawn = frame
        all_detected_strings = ["Errore nel processamento"]

    # Disegna gli FPS sul frame (già disegnato)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame_drawn, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Aggiorna GUI ---

    # Aggiorna lista a destra
    object_list.delete(0, tk.END)
    if not all_detected_strings:
        object_list.insert(tk.END, "Nessun oggetto rilevato")
    else:
        for obj_str in all_detected_strings:
            object_list.insert(tk.END, obj_str)

    # Aggiorna immagine webcam
    cv2image = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # Richiama questa funzione dopo 10 ms
    root.after(1, update_frame, root, lmain, cap, object_list, app, prev_time, fps)


if __name__ == '__main__':
    main()
