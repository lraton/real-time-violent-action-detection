import numpy as np
print("Loading saved data...")
loaded = np.load("models/violentvideo/violent_video1_cam1_person6.npz")
data = loaded["data"]
label = loaded["label"]
print("Data shape:", data.shape)
print("Label:", label)
print("Data:", data)
