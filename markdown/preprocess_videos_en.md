# Preprocess Videos Script

This script (`src/preprocess_videos.py`) automates the extraction of pose keypoints from video datasets using YOLOv8/11 and prepares them for LSTM-based action recognition.

## Code Overview

### Dependencies
- `multiprocessing`: For parallel processing of video directories.
- `ultralytics`: Uses the `yolo11n-pose.pt` model for pose tracking.
- `natsort`: For natural sorting of filenames.
- `numpy`, `torch`, `cv2`, `os`: Standard data processing and system libraries.

### Main Logic

1.  **Configuration**:
    -   Hardcoded list of `video_dirs` pointing to specific camera folders (`cam1` to `cam5`).
    -   Uses specific COCO keypoint indices for shoulders (indices 5 and 6) to normalize data.

2.  **Video Processing (`process_video_folder`)**:
    -   Scans folders for `.mp4`, `.avi`, `.mov` files.
    -   Assigns labels: `1` for folders with "violent" in the path, `0` for "non-violent".
    -   **Tracking**: Runs `model.track()` with `botsort.yaml` tracker configuration.
    -   **Normalization**:
        -   Calculates the center point between left and right shoulders.
        -   Calculates shoulder width as a scale factor.
        -   Normalizes all keypoints: `(point - center) / width`.
        -   Appends the confidence score to make `(x, y, conf)` triplets.
    -   **Filtering**:
        -   **Violent Videos**:
            -   Discards tracks with < 30 frames.
            -   Calculates a **Movement Score** (avg Euclidean distance magnitude of keypoint diffs).
            -   **Score < 1.5**: Classifies as **Spectator (Label 0)**.
            -   **Score >= 1.5**: Classifies as **Violent (Label 1)**.
        -   **Non-Violent Videos**:
            -   Discards tracks with <= 15 frames.
            -   Saves all remaining tracks as **Label 0**.

3.  **Data Saving (`save_keypoints_to_dataset`)**:
    -   Saves files to `../datasets/lstm_dataset/`.
    -   Naming convention: `{violent|nonviolent}_video{filename}_{cam}_person{id}.npz`.
    -   **Format**:
        -   `data`: NumPy float32 array of shape `(frames, 17, 3)`.
        -   `label`: Scalar integer (0 or 1).

4.  **Parallel Execution**:
    -   Uses `multiprocessing.Pool` to run folders in parallel.
    -   Workers count: `min(folders, cpu_count // 2)`.

### Helper Functions

-   `calculate_movement_score(frames)`: Computes the average frame-to-frame movement magnitude across all keypoints.
-   `normalize_keypoints_relative_to_torso(i, keypoints_normalized)`: Centers and scales keypoints relative to the torso. Includes fallback (center 0.5) if shoulders are not detected.
-   `video_display(...)`: (Optional/Commented) Displays video with bounding boxes and keypoints using `cv2`.

## Usage
Run the script directly:
```bash
python src/preprocess_videos.py
```
*Note: Ensure the model `../models/yolo11n-pose.pt` exists and video directories are correctly populated.*
