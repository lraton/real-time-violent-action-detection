from ultralytics import YOLO
import numpy as np
import cv2

# Carica il modello
model_pose = YOLO("yolo11n-pose.pt")

# Avvia il tracking sul video
results_generator = model_pose.track(
    '../video-dataset/violent/cam1/1.mp4', 
    tracker="botsort.yaml", 
    persist=True, 
    stream=True,
    verbose=False
)

# INDICI DEI KEYPOINT NEL FORMATO COCO (usato da YOLO)
# 5: spalla sinistra (left_shoulder)
# 6: spalla destra (right_shoulder)
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

# Itera sui risultati di ogni frame
for frame_result in results_generator:
    
    if frame_result.boxes.id is not None:
        
        person_ids = frame_result.boxes.id.cpu().numpy().astype(int)
        keypoints_normalized = frame_result.keypoints.xyn.cpu().numpy()

        for i, person_id in enumerate(person_ids):
            
            # --- FASE 1: Estrai i keypoint per la persona corrente ---
            person_keypoints_xyn = keypoints_normalized[i]
            
            # --- FASE 2: Calcola il punto centrale del corpo (anchor point) ---
            left_shoulder = person_keypoints_xyn[LEFT_SHOULDER_IDX]
            right_shoulder = person_keypoints_xyn[RIGHT_SHOULDER_IDX]
            
            # Calcola il punto medio tra le spalle
            # Procedi solo se le spalle sono state rilevate (coordinate > 0)
            if left_shoulder.sum() > 0 and right_shoulder.sum() > 0:
                center_point = (left_shoulder + right_shoulder) / 2
            else:
                # Fallback: se le spalle non sono visibili, usa il naso (indice 0) o salta il frame
                # Per semplicità, qui saltiamo la normalizzazione per questo frame se le spalle mancano
                center_point = np.array([0.0, 0.0]) # o continua con il frame successivo
                # continue 

            # --- FASE 3: Sottrai il punto centrale da tutti i keypoint ---
            # Questo sposta l'origine degli assi (0,0) al centro del torace della persona
            relative_keypoints = person_keypoints_xyn - center_point

            # --- ORA I TUOI DATI SONO PRONTI PER L'ALLENAMENTO ---
            print(f"ID Persona: {person_id}")
            print(f"Coordinate RELATIVE (rispetto al centro del torace):")
            print(relative_keypoints[:5]) # Stampa solo i primi 5 per brevità
            print("-" * 30)

            # Qui salveresti 'relative_keypoints' nel tuo dataset

    annotated_frame = frame_result.plot()
    display_frame = cv2.resize(annotated_frame, (1080,720), interpolation=cv2.INTER_AREA)
    # Mostra il frame nella finestra ridimensionabile
    cv2.imshow("YOLOv8 Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()