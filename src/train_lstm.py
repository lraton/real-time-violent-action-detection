import numpy as np
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

DATA_PATH = "models/violentvideo/"   # cartella dove hai salvato i file .npz
MODEL_PATH = "models/lstm_violence_detector.h5"
MAX_FRAMES = 150
BATCH_SIZE = 8
EPOCHS = 30

def main():
    X, y = [], []  # liste per dati e etichette

    print("Caricamento file .npz...")
    for file in glob.glob(os.path.join(DATA_PATH, "*.npz")):
        data = np.load(file)
        X.append(data["data"])
        y.append(int(data["label"]))
        print(f"Caricato {os.path.basename(file)}")

    print(f"Totale sequenze: {len(X)}")

    X, y = normalize_keypoints(X, y)
    create_and_train_model(X, y)

    print("\n Training completato!")

def normalize_keypoints(X, y): #
    print("Normalizzazione delle sequenze...")
    X = pad_sequences(X, maxlen=MAX_FRAMES, dtype='float32', padding='post', truncating='post')
    X = X.reshape(X.shape[0], X.shape[1], -1)
    y = np.array(y, dtype=np.float32)
    print(f"Forma finale X: {X.shape}, y: {y.shape}")
    return X, y

def create_and_train_model(X, y):
    # Suddivisione in training e validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Dati training: {X_train.shape}, validation: {X_val.shape}")

    # Creazione del modello LSTM
    print("Creazione modello LSTM...")
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, X.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    print("Avvio dell'addestramento...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

if __name__ == '__main__':
    main()
