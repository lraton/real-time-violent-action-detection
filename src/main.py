import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import time
import math
from violence_detection_system import ViolenceDetectionSystem

class VideoStream:
    # Oggetto per gestire ogni stream video

    def __init__(self, video_path, stream_id):
        self.video_path = video_path
        self.stream_id = stream_id
        self.cap = cv2.VideoCapture(video_path)
        self.counter_frames = 0
        self.old_frame = None
        self.old_all_detected_strings = []
        self.prev_time = time.time()
        self.fps = 0

    def release(self):
        if self.cap is not None:
            self.cap.release()


class UI:

    def __init__(self, root):
        self.root = root
        self.root.title("VIOLENCE DETECTION SYSTEM")
        self.root.geometry("1600x900")
        self.root.configure(bg="#0a0e27")

        # Variabili di controllo
        self.current_frame_skip = 2  # Valore predefinito
        self.frame_skip_var = tk.IntVar(value=2) # Variabile Tkinter per lo spinbox

        # Scheme colori
        # Backgrounds
        self.bg_dark = "#2b2b2b"  # Sfondo principale (Grigio Scuro Antracite)
        self.bg_panel = "#333333"  # Pannelli laterali (Grigio leggermente più chiaro)
        self.bg_video = "#1e1e1e"  # Sfondo contenitore video (Quasi nero)
        self.bg_header = "#3c3c3c"  # Sfondo header video

        # Accenti
        self.accent_cyan = "#5c9aff"  # Blu Acciaio (Titoli, bottoni principali)
        self.accent_purple = "#757575"  # Grigio Medio (Separatori, dettagli secondari)
        self.accent_red = "#d32f2f"  # Rosso standard "Danger" (non fluo)

        # Testo e Indicatori
        self.text_color = "#ffffff"  # Bianco puro per leggibilità
        self.text_dim = "#cccccc"  # Grigio chiaro per testo secondario
        self.success_color = "#388e3c"  # Verde scuro professionale

        self.video_streams = []  # Lista di oggetti VideoStream
        self.video_labels = []  # Lista di etichette per la visualizzazione video
        self.fps_labels = []  # Lista di etichette FPS
        self.app = None  # Istanza della tua app di rilevamento
        self.is_processing = False

        self.create_widgets()

    def create_widgets(self):
        # ========== HEADER ==========
        header = tk.Frame(self.root, bg=self.bg_dark, height=80)
        header.pack(fill="x", padx=0, pady=0)

        title_label = tk.Label(header, text="MULTI-STREAM VIOLENCE DETECTION", font=("Courier New", 24, "bold"), fg=self.accent_cyan, bg=self.bg_dark)
        title_label.pack(side="left", padx=30, pady=15)

        # Indicatore di stato
        self.status_indicator = tk.Label(header, text="● STANDBY", font=("Courier New", 14, "bold"), fg="#888888", bg=self.bg_dark)
        self.status_indicator.pack(side="right", padx=30, pady=15)

        # Linea separatrice
        separator = tk.Frame(self.root, bg=self.accent_purple, height=2)
        separator.pack(fill="x")

        # ========== MAIN CONTAINER ==========
        main_container = tk.Frame(self.root, bg=self.bg_dark)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # ========== LEFT PANEL - VIDEO GRID ==========
        self.left_panel = tk.Frame(main_container, bg=self.bg_panel, relief="flat", bd=0)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Left panel header
        left_header = tk.Frame(self.left_panel, bg=self.bg_panel)
        left_header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(left_header, text="► VIDEO FEEDS", font=("Courier New", 14, "bold"), fg=self.accent_cyan, bg=self.bg_panel).pack(side="left")

        self.stream_count_label = tk.Label(left_header, text="0 streams", font=("Courier New", 12, "bold"), fg=self.success_color, bg=self.bg_panel)
        self.stream_count_label.pack(side="right")

        # Video grid container
        self.video_grid_frame = tk.Frame(self.left_panel, bg=self.bg_video, relief="flat")
        self.video_grid_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Messaggio iniziale vuoto
        self.empty_label = tk.Label(self.video_grid_frame,
                                    bg=self.bg_video,
                                    text="NO VIDEOS LOADED\n\n▼\n\nClick 'LOAD VIDEOS' to start",
                                    font=("Courier New", 16, "bold"),
                                    fg="#444444")
        self.empty_label.pack(fill="both", expand=True)

        # ========== RIGHT PANEL - DETECTION LIST ==========
        right_panel = tk.Frame(main_container, bg=self.bg_panel, relief="flat", bd=0, width=450)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)

        # Right panel header
        right_header = tk.Frame(right_panel, bg=self.bg_panel)
        right_header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(right_header, text="► DETECTION DATA", font=("Courier New", 14, "bold"), fg=self.accent_purple, bg=self.bg_panel).pack(side="left")

        # Contatore rilevazioni
        self.detection_count = tk.Label(right_header, text="0", font=("Courier New", 12, "bold"), fg=self.accent_red, bg=self.bg_panel)
        self.detection_count.pack(side="right")

        # Contenitore lista con scrollbar
        list_container = tk.Frame(right_panel, bg=self.bg_video)
        list_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        scrollbar = tk.Scrollbar(list_container, bg=self.bg_video)
        scrollbar.pack(side="right", fill="y")

        self.object_listbox = tk.Listbox(list_container,
                                         bg=self.bg_video,
                                         fg=self.text_color,
                                         font=("Courier New", 10),
                                         relief="flat",
                                         selectmode=tk.SINGLE,
                                         highlightthickness=0,
                                         selectbackground=self.accent_purple,
                                         selectforeground="white",
                                         activestyle="none",
                                         yscrollcommand=scrollbar.set)
        self.object_listbox.pack(fill="both", expand=True, padx=2, pady=2)
        scrollbar.config(command=self.object_listbox.yview)

        # Messaggio iniziale
        self.object_listbox.insert(tk.END, "╔═══════════════════════════════╗")
        self.object_listbox.insert(tk.END, "║   WAITING FOR VIDEO INPUT...  ║")
        self.object_listbox.insert(tk.END, "╚═══════════════════════════════╝")

        # ========== FOOTER - CONTROLS ==========
        footer = tk.Frame(self.root, bg=self.bg_panel, height=80)
        footer.pack(fill="x", padx=20, pady=(0, 20))

        controls_frame = tk.Frame(footer, bg=self.bg_panel)
        controls_frame.pack(expand=True, pady=20)

        # --- SEZIONE FRAME SKIP ---
        skip_frame_container = tk.Frame(controls_frame, bg=self.bg_panel, bd=1, relief="solid")
        skip_frame_container.pack(side="left", padx=(0, 20), pady=10)
        
        tk.Label(skip_frame_container, text="FRAME SKIP:", font=("Courier New", 10, "bold"), fg=self.text_dim, bg=self.bg_panel).pack(side="left", padx=5)
        
        # Spinbox per selezionare 1, 2, 3, 4, 5
        self.skip_spinner = tk.Spinbox(skip_frame_container, 
                                       from_=1, to=5, 
                                       textvariable=self.frame_skip_var, 
                                       width=3, 
                                       font=("Courier New", 12, "bold"),
                                       state="readonly", # L'utente può cliccare le frecce ma non scrivere
                                       bg="white")
        self.skip_spinner.pack(side="left", padx=5, pady=5)
        # --------------------------

        # Pulsante Carica Video
        self.load_btn = tk.Button(controls_frame,
                                  text="▶ LOAD VIDEOS",
                                  font=("Courier New", 12, "bold"),
                                  bg=self.accent_cyan,
                                  fg=self.bg_dark,
                                  activebackground=self.accent_purple,
                                  activeforeground="white",
                                  relief="flat",
                                  padx=30,
                                  pady=12,
                                  cursor="hand2",
                                  command=self.load_videos)
        self.load_btn.pack(side="left", padx=10)

        # Pulsante Stop
        self.stop_btn = tk.Button(controls_frame,
                                  text="■ STOP ALL",
                                  font=("Courier New", 12, "bold"),
                                  bg=self.accent_red,
                                  fg="white",
                                  activebackground="#cc0044",
                                  activeforeground="white",
                                  relief="flat",
                                  padx=30,
                                  pady=12,
                                  cursor="hand2",
                                  state="disabled",
                                  command=self.stop_videos)
        self.stop_btn.pack(side="left", padx=10)

        # Pulsante cartella sospetti
        self.folder_btn = tk.Button(controls_frame,
                                    text="📁 Open Suspect Folder",
                                    font=("Courier New", 12, "bold"),
                                    bg=self.accent_cyan,
                                    fg=self.bg_dark,
                                    activebackground=self.accent_purple,
                                    activeforeground="white",
                                    relief="flat",
                                    padx=30,
                                    pady=12,
                                    cursor="hand2",
                                    state="disabled",
                                    command=self.suspect_folder)
        self.folder_btn.pack(side="left", padx=10)
        self.folder_btn.config(state="normal")

        # Pulsante cartella LOGS
        self.logs_btn = tk.Button(controls_frame,
                                  text="📁 Open LOGS Folder",
                                  font=("Courier New", 12, "bold"),
                                  bg=self.accent_cyan,
                                  fg=self.bg_dark,
                                  activebackground=self.accent_purple,
                                  activeforeground="white",
                                  relief="flat",
                                  padx=30,
                                  pady=12,
                                  cursor="hand2",
                                  state="disabled",
                                  command=self.logs_folder)
        self.logs_btn.pack(side="left", padx=10)
        self.logs_btn.config(state="normal")

    def load_videos(self):
        # Apri file dialog per selezionare video
        video_paths = filedialog.askopenfilenames(title="Select Video Files (Multiple Selection)", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])

        if video_paths:
            self.start_videos(video_paths)

    def calculate_grid_layout(self, num_videos):
        # Calcola il layout della griglia in base al numero di video

        if num_videos == 1:
            return 1, 1
        elif num_videos == 2:
            return 1, 2  # 1 row, 2 columns
        elif num_videos == 3:
            return 2, 2  # 2 rows, 2 columns (one empty)
        elif num_videos == 4:
            return 2, 2  # 2x2 grid
        elif num_videos <= 6:
            return 2, 3  # 2 rows, 3 columns
        elif num_videos <= 9:
            return 3, 3  # 3x3 grid
        else:
            cols = math.ceil(math.sqrt(num_videos))
            rows = math.ceil(num_videos / cols)
            return rows, cols

    def create_video_grid(self, num_videos):
        # Crea il layout della griglia per la visualizzazione dei video
        # Pulisci la griglia esistente
        for widget in self.video_grid_frame.winfo_children():
            widget.destroy()

        self.video_labels = []
        self.fps_labels = []

        if num_videos == 0:
            self.empty_label = tk.Label(self.video_grid_frame, bg=self.bg_video, text="NO VIDEOS LOADED", font=("Courier New", 16, "bold"), fg="#444444")
            self.empty_label.pack(fill="both", expand=True)
            return

        rows, cols = self.calculate_grid_layout(num_videos)

        # Configura i pesi della griglia per una distribuzione uniforme
        for i in range(rows):
            self.video_grid_frame.rowconfigure(i, weight=1)
        for j in range(cols):
            self.video_grid_frame.columnconfigure(j, weight=1)

        # Crea le celle video
        for idx in range(num_videos):
            row = idx // cols
            col = idx % cols

            # Cell container
            cell = tk.Frame(self.video_grid_frame, bg=self.bg_video, relief="solid", bd=1)
            cell.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)

            # Intestazione video con ID stream e FPS
            header = tk.Frame(cell, bg="#0d1120", height=30)
            header.pack(fill="x")
            header.pack_propagate(False)

            tk.Label(header, text=f"STREAM #{idx+1}", font=("Courier New", 9, "bold"), fg=self.accent_cyan, bg="#0d1120").pack(side="left", padx=10, pady=5)

            fps_label = tk.Label(header, text="FPS: 0.0", font=("Courier New", 9, "bold"), fg=self.success_color, bg="#0d1120")
            fps_label.pack(side="right", padx=10, pady=5)
            self.fps_labels.append(fps_label)

            # Visualizzazione video
            video_label = tk.Label(cell, bg="#000000")
            video_label.pack(fill="both", expand=True)
            self.video_labels.append(video_label)

    def start_videos(self, video_paths):
        # Inizializza le acquisizioni video e avvia l'elaborazione
        # Ferma eventuali stream esistenti
        self.stop_videos()

        # LEGGI IL VALORE DEL FRAME SKIP DALLA UI
        self.current_frame_skip = self.frame_skip_var.get()
        print(f"Starting videos with Frame Skip: {self.current_frame_skip}")

        # Crea stream video
        for idx, path in enumerate(video_paths):
            stream = VideoStream(path, idx)
            if stream.cap.isOpened():
                self.video_streams.append(stream)
            else:
                print(f"Could not open video: {path}")
                stream.release()

        if not self.video_streams:
            self.object_listbox.delete(0, tk.END)
            self.object_listbox.insert(tk.END, "✗ ERROR: Could not open any video files")
            return

        # Crea il layout della griglia
        self.create_video_grid(len(self.video_streams))

        # Aggiorna lo stato dell'interfaccia utente
        self.load_btn.config(state="disabled")
        self.skip_spinner.config(state="disabled") 
        self.stop_btn.config(state="normal")
        
        self.status_indicator.config(text=f"● PROCESS (SKIP {self.current_frame_skip})", fg=self.success_color)
        self.stream_count_label.config(text=f"{len(self.video_streams)} streams")

        # Pulisci la lista delle rilevazioni  
        self.object_listbox.delete(0, tk.END)
        self.object_listbox.insert(tk.END, "╔═══════════════════════════════╗")
        self.object_listbox.insert(tk.END, "║     DETECTION INITIALIZED     ║")
        self.object_listbox.insert(tk.END, "╚═══════════════════════════════╝")

        self.is_processing = True

        # Avvia il ciclo di aggiornamento dei frame
        self.update_frames()

    def stop_videos(self):
        # Ferma tutti gli stream video e resetta l'interfaccia utente
        self.is_processing = False

        for stream in self.video_streams:
            stream.release()
        self.video_streams = []

        self.load_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_indicator.config(text="● STANDBY", fg="#888888")
        self.stream_count_label.config(text="0 streams")

        # Resetta il contatore delle rilevazioni
        self.detection_count.config(text="0")

        # Pulisci la griglia
        self.create_video_grid(0)

        self.object_listbox.delete(0, tk.END)
        self.object_listbox.insert(tk.END, "╔═══════════════════════════════╗")
        self.object_listbox.insert(tk.END, "║      VIDEO PROCESSING END     ║")
        self.object_listbox.insert(tk.END, "╚═══════════════════════════════╝")

    def suspect_folder(self):
        # Apri la cartella suspect nell'explorer
        import os
        suspect_folder_path = os.path.abspath("../suspect/")
        if not os.path.exists(suspect_folder_path):
            os.makedirs(suspect_folder_path)
        os.startfile(suspect_folder_path)

    def logs_folder(self):
        # Apri la cartella logs nell'explorer
        import os
        logs_folder_path = os.path.abspath("logs/")
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
        os.startfile(logs_folder_path)

    def update_frames(self):
        # Ciclo principale di elaborazione dei frame per tutti gli stream
        if not self.is_processing or not self.video_streams:
            return

        all_detections = []
        active_streams = []

        # Usa la variabile di istanza catturata all'avvio
        active_frame_skip = self.current_frame_skip

        for idx, stream in enumerate(self.video_streams):
            if not stream.cap.isOpened():
                continue

            active_streams.append(stream)

            # Reset counter_frames ogni frame_skip frame 
            if stream.counter_frames == active_frame_skip:
                stream.counter_frames = 0

            # Calcolo FPS
            current_time = time.time()
            elapsed_time = current_time - stream.prev_time
            stream.prev_time = current_time
            if elapsed_time > 0:
                stream.fps = 1.0 / elapsed_time

            success, frame = stream.cap.read()

            if not success:
                print(f"Stream #{idx+1} ended")
                continue

            try:
                # Processa il frame solo ogni 'frame_skip' frame
                if stream.counter_frames % active_frame_skip == 0:
                    if self.app is not None:
                        frame_drawn, detected_strings = self.app.process_frame(frame, active_frame_skip)
                    else:
                        # Modalità demo senza app
                        frame_drawn = frame.copy()
                        detected_strings = [
                            f"[Stream #{idx+1}] Person detected - Non-violent",
                        ]
                    stream.old_frame = frame_drawn
                    stream.old_all_detected_strings = detected_strings
                else:
                    frame_drawn = stream.old_frame if stream.old_frame is not None else frame
                    detected_strings = stream.old_all_detected_strings if stream.old_all_detected_strings else []
            except Exception as e:
                print(f"Error processing stream #{idx+1}: {e}")
                frame_drawn = frame
                detected_strings = [f"[Stream #{idx+1}] ✗ Processing error"]

            # Raccogli tutte le rilevazioni
            all_detections.extend(detected_strings)

            # Aggiorna l'etichetta FPS
            if idx < len(self.fps_labels):
                self.fps_labels[idx].config(text=f"FPS: {stream.fps:.1f}")

            # Calcola la dimensione di visualizzazione basata sulla griglia
            if idx < len(self.video_labels):
                label_width = self.video_labels[idx].winfo_width()
                label_height = self.video_labels[idx].winfo_height()

                # Usa valori predefiniti ragionevoli se la finestra non è completamente renderizzata
                if label_width < 10:
                    label_width = 400
                if label_height < 10:
                    label_height = 300

                frame_drawn = cv2.resize(frame_drawn, (label_width, label_height))

                # Aggiorna la visualizzazione video
                cv2image = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_labels[idx].imgtk = imgtk
                self.video_labels[idx].configure(image=imgtk)

            # Incrementa il contatore dei frame
            stream.counter_frames += 1

        # Aggiorna la lista delle rilevazioni
        self.object_listbox.delete(0, tk.END)
        if not all_detections:
            self.object_listbox.insert(tk.END, "[ NO DETECTION ]")
        else:
            self.detection_count.config(text=str(len(all_detections)))

            for obj_str in all_detections:
                self.object_listbox.insert(tk.END, obj_str)
                # Codifica colore per rilevazioni violente   
                if "violent" in obj_str.lower() or "knife" in obj_str.lower() or "weapon" in obj_str.lower():
                    self.object_listbox.itemconfig(tk.END, fg=self.accent_red)

        # Controlla se tutti gli stream sono terminati
        if not active_streams:
            print("All streams ended")
            self.stop_videos()
            return

        # Continue loop
        self.root.after(1, self.update_frames)


def main():
    root = tk.Tk()
    app = UI(root)

    app.app = ViolenceDetectionSystem(knife_model_path="../models/knife/run2/weights/best.pt",
                                      pose_model_path="../models/yolo11n-pose.pt",
                                      lstm_model_path="../models/lstm_violence_detector_v8.keras")

    root.mainloop()


if __name__ == '__main__':
    main()
