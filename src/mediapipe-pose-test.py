import mediapipe as mp
import cv2

# --- CHANGED LINES ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose  # Import the Pose model
# --- END CHANGED LINES ---
backup_video_path = '../video-dataset/violent/cam1/37.mp4'
cap = cv2.VideoCapture(backup_video_path) # or your video_path

# --- CHANGED BLOCK ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image) # Process with pose model
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        # --- END CHANGED BLOCK ---

        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()