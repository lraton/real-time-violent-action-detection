import numpy as np
print("Loading saved data...")
loaded = np.load("../datasets/lstm_dataset/nonviolent_video5_cam1_person1.npz")
data = loaded["data"]
label = loaded["label"]
print("Data shape:", data.shape)
print("Label:", label)
print("Data:", data)
