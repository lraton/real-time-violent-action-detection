import cv2
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from violence_detection_system import ViolenceDetectionSystem

counter_frames = 0  # Contatore dei frame processati
old_frame = None  # Memorizza l'ultimo frame processato
old_all_detected_strings = []  # Memorizza l'ultimo risultato di rilevamento
frame_skip = 2  # Numero di frame da saltare prima di un nuovo processamento


def main():
    prev_time = 0  # Tempo del frame precedente
    fps = 0  # Valore FPS calcolato
    camera_index = 0  # Modifica questo indice se necessario (spesso 0 o 1)
    backup_video_path = '../video-dataset/violent/cam2/104.mp4'

    # Inizializza la classe che gestisce YOLO
    app = ViolenceDetectionSystem(knife_model_path="../models/knife/run2/weights/best.pt",
                                  pose_model_path="../models/yolo11n-pose.pt",
                                  lstm_model_path="../models/lstm_violence_detector_v6.keras")

    # --- Tkinter GUI ---
    root = tk.Tk()
    root.title("YOLO Pose + Object Detection + LSTM GUI")

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

    cap.set(3, 256)
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
    global counter_frames
    global old_frame
    global old_all_detected_strings
    global frame_skip
    # Reset counter_frames ogni 4 frame
    if counter_frames == frame_skip:
        counter_frames = 0

    # Calcolo FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 0:
        fps = 1.0 / elapsed_time
    prev_time = current_time  # Salva il tempo corrente per la prossima iterazione

    success, frame = cap.read()
    #fps = cap.get(cv2.CAP_PROP_FPS) # Ottieni FPS del video/camera
    if not success:
        print("Fine del video o errore nella lettura del frame.")
        # Chiudi tutto
        cap.release()
        root.quit()  # Chiude il mainloop di Tkinter
        root.destroy()  # Distrugge la finestra
        return
    try:  # Processa il frame solo ogni 'frame_skip' frame
        if counter_frames % frame_skip == 0:
            #print('Processo nuovo frame')
            frame_drawn, all_detected_strings = app.process_frame(frame, frame_skip)
            old_frame = frame_drawn
            old_all_detected_strings = all_detected_strings
        else:  # Usa l'ultimo frame processato
            #print('Uso frame vecchio')
            #frame_drawn = frame  #usa il frame originale per evitare scatti ma senza disegni
            frame_drawn = old_frame if old_frame is not None else frame  # Usa l'ultimo frame processato se disponibile ma piu scattoso
            all_detected_strings = old_all_detected_strings if old_all_detected_strings else ["Nessun oggetto rilevato"]
    except Exception as e:
        print(f"Errore durante process_frame: {e}")
        # Continua a mostrare il frame originale in caso di errore
        frame_drawn = frame
        all_detected_strings = ["Errore nel processamento"]

    # Disegna gli FPS sul frame (già disegnato)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame_drawn, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Ridimensiona il frame per la visualizzazione nella GUI
    display_width = 640  # Larghezza desiderata
    display_height = 480  # Altezza desiderata
    frame_drawn = cv2.resize(frame_drawn, (display_width, display_height))

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

    # Incrementa il contatore dei frame
    counter_frames += 1

    # Richiama questa funzione dopo 10 ms
    root.after(1, update_frame, root, lmain, cap, object_list, app, prev_time, fps)


if __name__ == '__main__':
    main()