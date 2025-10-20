import numpy as np
print("Loading saved data...")
loaded = np.load("models/violentvideo/video1_person2.npz")
data = loaded["data"]
label = loaded["label"]
print("Data shape:", data.shape)
print("Label:", label)
print("Data:", data)
