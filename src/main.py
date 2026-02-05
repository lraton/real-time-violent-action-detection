import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import time
from violence_detection_system import ViolenceDetectionSystem

class VideoStream:
    # Oggetto per gestire il singolo stream video
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.counter_frames = 0
        self.old_frame = None
        self.old_all_detected_strings = []
        self.prev_time = time.time()
        self.fps = 0
        self.font_title = ("Courier New", 28, "bold") # Era 24
        self.font_header = ("Courier New", 18, "bold") # Era 14
        self.font_body = ("Courier New", 14, "bold")   # Era 12/10

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
        self.current_frame_skip = 2  
        self.frame_skip_var = tk.IntVar(value=2) 

        # Scheme colori
        self.bg_dark = "#2b2b2b"
        self.bg_panel = "#333333"
        self.bg_video = "#1e1e1e"
        self.bg_header = "#3c3c3c"

        self.accent_cyan = "#5c9aff"
        self.accent_purple = "#757575"
        self.accent_red = "#d32f2f"

        self.text_color = "#ffffff"
        self.text_dim = "#cccccc"
        self.success_color = "#388e3c"

        # Variabili Stream
        self.video_stream = None 
        self.video_label = None
        self.fps_label = None 
        
        self.app = None  
        self.is_processing = False

        self.create_widgets()

    def create_widgets(self):
        # ========== HEADER ==========
        header = tk.Frame(self.root, bg=self.bg_dark, height=80)
        header.pack(fill="x", padx=0, pady=0)

        title_label = tk.Label(header, text="VIOLENCE DETECTION", font=("Courier New", 24, "bold"), fg=self.accent_cyan, bg=self.bg_dark)
        title_label.pack(side="left", padx=30, pady=15)

        # Indicatore di stato
        self.status_indicator = tk.Label(header, text="● STANDBY", font=("Courier New", 14, "bold"), fg="#888888", bg=self.bg_dark)
        self.status_indicator.pack(side="right", padx=30, pady=15)

        separator = tk.Frame(self.root, bg=self.accent_purple, height=2)
        separator.pack(fill="x")

        # ========== MAIN CONTAINER ==========
        main_container = tk.Frame(self.root, bg=self.bg_dark)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # ========== LEFT PANEL - VIDEO DISPLAY ==========
        self.left_panel = tk.Frame(main_container, bg=self.bg_panel, relief="flat", bd=0)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Left panel header
        left_header = tk.Frame(self.left_panel, bg=self.bg_panel)
        left_header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(left_header, text="► LIVE FEED", font=("Courier New", 14, "bold"), fg=self.accent_cyan, bg=self.bg_panel).pack(side="left")

        # Video container
        self.video_container = tk.Frame(self.left_panel, bg=self.bg_video, relief="flat")
        self.video_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.video_container.pack_propagate(False)

        # Messaggio iniziale / Placeholder
        self.empty_label = tk.Label(self.video_container,
                                    bg=self.bg_video,
                                    text="NO VIDEO LOADED\n\n▼\n\nClick 'LOAD VIDEO' to start",
                                    font=("Courier New", 16, "bold"),
                                    fg="#444444")
        self.empty_label.pack(fill="both", expand=True)

        # ========== RIGHT PANEL - DETECTION LIST ==========
        right_panel = tk.Frame(main_container, bg=self.bg_panel, relief="flat", bd=0, width=450)
        right_panel.pack(side="right", fill="both", padx=(10, 0))
        right_panel.pack_propagate(False)

        right_header = tk.Frame(right_panel, bg=self.bg_panel)
        right_header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(right_header, text="► DETECTION DATA", font=("Courier New", 14, "bold"), fg=self.accent_purple, bg=self.bg_panel).pack(side="left")

        self.detection_count = tk.Label(right_header, text="0", font=("Courier New", 12, "bold"), fg=self.accent_red, bg=self.bg_panel)
        self.detection_count.pack(side="right")

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

        self.object_listbox.insert(tk.END, "╔═══════════════════════════════╗")
        self.object_listbox.insert(tk.END, "║   WAITING FOR VIDEO INPUT...  ║")
        self.object_listbox.insert(tk.END, "╚═══════════════════════════════╝")

        # ========== FOOTER - CONTROLS ==========
        footer = tk.Frame(self.root, bg=self.bg_panel, height=80)
        footer.pack(fill="x", padx=20, pady=(0, 20))

        controls_frame = tk.Frame(footer, bg=self.bg_panel)
        controls_frame.pack(expand=True, pady=20)

        # Frame Skip
        skip_frame_container = tk.Frame(controls_frame, bg=self.bg_panel, bd=1, relief="solid")
        skip_frame_container.pack(side="left", padx=(0, 20), pady=10)
        
        tk.Label(skip_frame_container, text="FRAME SKIP:", font=("Courier New", 10, "bold"), fg=self.text_dim, bg=self.bg_panel).pack(side="left", padx=5)
        
        self.skip_spinner = tk.Spinbox(skip_frame_container, 
                                       from_=1, to=5, 
                                       textvariable=self.frame_skip_var, 
                                       width=3, 
                                       font=("Courier New", 12, "bold"),
                                       state="readonly",
                                       bg="white")
        self.skip_spinner.pack(side="left", padx=5, pady=5)

        # Pulsante Carica Video
        self.load_btn = tk.Button(controls_frame,
                                  text="▶ LOAD VIDEO",
                                  font=("Courier New", 12, "bold"),
                                  bg=self.accent_cyan,
                                  fg=self.bg_dark,
                                  activebackground=self.accent_purple,
                                  activeforeground="white",
                                  relief="flat",
                                  padx=30,
                                  pady=12,
                                  cursor="hand2",
                                  command=self.load_video)
        self.load_btn.pack(side="left", padx=10)

        # Pulsante Stop
        self.stop_btn = tk.Button(controls_frame,
                                  text="■ STOP",
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
                                  command=self.stop_video)
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
                                    state="normal",
                                    command=self.suspect_folder)
        self.folder_btn.pack(side="left", padx=10)

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
                                  state="normal",
                                  command=self.logs_folder)
        self.logs_btn.pack(side="left", padx=10)

    def load_video(self):
        # Selezione file singolo
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])

        if video_path:
            self.start_video(video_path)

    def setup_video_display(self):
        # Prepara il contenitore per un video
        for widget in self.video_container.winfo_children():
            widget.destroy()

        # Intestazione video (Stream info e FPS)
        header = tk.Frame(self.video_container, bg="#0d1120", height=30)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="SOURCE: LOCAL FILE", font=("Courier New", 10, "bold"), fg=self.accent_cyan, bg="#0d1120").pack(side="left", padx=10, pady=5)

        self.fps_label = tk.Label(header, text="FPS: 0.0", font=("Courier New", 10, "bold"), fg=self.success_color, bg="#0d1120")
        self.fps_label.pack(side="right", padx=10, pady=5)

        # Visualizzazione video principale
        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack(fill="both", expand=True)

    def start_video(self, video_path):
        # Ferma eventuale video precedente
        self.stop_video()

        self.current_frame_skip = self.frame_skip_var.get()
        print(f"Starting video with Frame Skip: {self.current_frame_skip}")

        # Crea singolo stream
        self.video_stream = VideoStream(video_path)
        
        if not self.video_stream.cap.isOpened():
            print(f"Could not open video: {video_path}")
            self.video_stream.release()
            self.video_stream = None
            self.object_listbox.delete(0, tk.END)
            self.object_listbox.insert(tk.END, "✗ ERROR: Could not open video file")
            return

        # Prepara la UI
        self.setup_video_display()

        # Aggiorna stato bottoni
        self.load_btn.config(state="disabled")
        self.skip_spinner.config(state="disabled") 
        self.stop_btn.config(state="normal")
        
        self.status_indicator.config(text=f"● PROCESSING (SKIP {self.current_frame_skip})", fg=self.success_color)

        self.object_listbox.delete(0, tk.END)
        self.object_listbox.insert(tk.END, "╔═══════════════════════════════╗")
        self.object_listbox.insert(tk.END, "║     DETECTION INITIALIZED     ║")
        self.object_listbox.insert(tk.END, "╚═══════════════════════════════╝")

        self.is_processing = True
        self.update_frames()

    def stop_video(self):
        self.is_processing = False

        if self.video_stream:
            self.video_stream.release()
            self.video_stream = None

        self.load_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.skip_spinner.config(state="normal") 
        self.status_indicator.config(text="● STANDBY", fg="#888888")

        self.detection_count.config(text="0")

        # Pulisci contenitore video
        for widget in self.video_container.winfo_children():
            widget.destroy()
        
        self.empty_label = tk.Label(self.video_container, bg=self.bg_video, text="NO VIDEO LOADED", font=("Courier New", 16, "bold"), fg="#444444")
        self.empty_label.pack(fill="both", expand=True)

        self.object_listbox.delete(0, tk.END)
        self.object_listbox.insert(tk.END, "╔═══════════════════════════════╗")
        self.object_listbox.insert(tk.END, "║      VIDEO PROCESSING END     ║")
        self.object_listbox.insert(tk.END, "╚═══════════════════════════════╝")

    def suspect_folder(self):
        import os
        suspect_folder_path = os.path.abspath("suspect/")
        if not os.path.exists(suspect_folder_path):
            os.makedirs(suspect_folder_path)
        os.startfile(suspect_folder_path)

    def logs_folder(self):
        import os
        logs_folder_path = os.path.abspath("logs/")
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
        os.startfile(logs_folder_path)

    def update_frames(self):
        if not self.is_processing or self.video_stream is None:
            return

        stream = self.video_stream
        
        if not stream.cap.isOpened():
            self.stop_video()
            return

        # Reset counter
        if stream.counter_frames == self.current_frame_skip:
            stream.counter_frames = 0

        # Calcolo FPS
        current_time = time.time()
        elapsed_time = current_time - stream.prev_time
        stream.prev_time = current_time
        if elapsed_time > 0:
            stream.fps = 1.0 / elapsed_time

        success, frame = stream.cap.read()

        if not success:
            print("Video ended")
            self.stop_video()
            return

        try:
            # Processing Frame
            if stream.counter_frames % self.current_frame_skip == 0:
                if self.app is not None:
                    frame_drawn, detected_strings = self.app.process_frame(frame, self.current_frame_skip)
                else:
                    # Modalità demo
                    frame_drawn = frame.copy()
                    detected_strings = ["Person detected - Non-violent"]
                
                stream.old_frame = frame_drawn
                stream.old_all_detected_strings = detected_strings
            else:
                frame_drawn = stream.old_frame if stream.old_frame is not None else frame
                detected_strings = stream.old_all_detected_strings if stream.old_all_detected_strings else []

        except Exception as e:
            print(f"Error processing video: {e}")
            frame_drawn = frame
            detected_strings = ["✗ Processing error"]

        # Update FPS Label
        if self.fps_label:
            self.fps_label.config(text=f"FPS: {stream.fps:.1f}")

        # Resize e Update Video Image
        if self.video_label:
            container_w = self.video_container.winfo_width()
            container_h = self.video_container.winfo_height() - 40 # Sottraiamo lo spazio dell'header FPS

            if container_w < 100: container_w = 800 # Fallback iniziale
            if container_h < 100: container_h = 600

            # Opzionale: Calcola aspect ratio per non schiacciare l'immagine
            h, w = frame_drawn.shape[:2]
            aspect = w / h
            
            if container_w / container_h > aspect:
                new_h = container_h
                new_w = int(aspect * new_h)
            else:
                new_w = container_w
                new_h = int(new_w / aspect)

            frame_resized = cv2.resize(frame_drawn, (new_w, new_h))
            
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        stream.counter_frames += 1

        # Aggiorna lista rilevazioni
        self.object_listbox.delete(0, tk.END)
        if not detected_strings:
            self.object_listbox.insert(tk.END, "[ NO DETECTION ]")
        else:
            self.detection_count.config(text=str(len(detected_strings)))
            for obj_str in detected_strings:
                self.object_listbox.insert(tk.END, obj_str)
                if "violent" in obj_str.lower() or "knife" in obj_str.lower() or "weapon" in obj_str.lower():
                    self.object_listbox.itemconfig(tk.END, fg=self.accent_red)

        # Loop
        self.root.after(1, self.update_frames)

def main():
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 2.0)
    app = UI(root)

    app.app = ViolenceDetectionSystem(knife_model_path="../models/knife/run2/weights/best.pt",
                                      pose_model_path="../models/yolo11n-pose.pt",
                                      lstm_model_path="../models/lstm_violence_detector_v8.keras")

    root.mainloop()

if __name__ == '__main__':
    main()