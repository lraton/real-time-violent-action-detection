import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disabilita ottimizzazioni OneDNN per evitare problemi di compatibilità
import numpy as np
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

DATA_PATH = "../models/lstm_dataset/"  # cartella dove hai salvato i file .npz
MODEL_PATH = "../models/lstm_violence_detector_v6.keras"
MAX_FRAMES = 150
BATCH_SIZE = 8
EPOCHS = 100


def main():
    X, y = [], []  # liste per dati e etichette

    print("Caricamento file .npz...")
    for file in glob.glob(os.path.join(DATA_PATH, "*.npz")):
        data = np.load(file)
        X.append(data["data"])  # sequenza di keypoints
        y.append(int(data["label"]))  # violent=1, non-violent=0
        #print(f"Caricato {os.path.basename(file)}")

    print(f"Totale sequenze: {len(X)}")

    X, y = normalize_keypoints(X, y)
    create_and_train_model(X, y)

    print("\n Training completato!")


def normalize_keypoints(X, y):  # Normalizza le sequenze di keypoints
    print("Normalizzazione delle sequenze...")
    X = pad_sequences(X, maxlen=MAX_FRAMES, dtype='float32', padding='post', truncating='post')
    X = X.reshape(X.shape[0], X.shape[1], -1)
    y = np.array(y, dtype=np.float32)
    print(f"Forma finale X: {X.shape}, y: {y.shape}")
    return X, y


def create_and_train_model(X, y):
    #Separiamo il TEST SET (15% del totale)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Separiamo TRAINING e VALIDATION dai dati rimanenti (20% del rimanente)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42
    )

    print(f"Dataset Split:")
    print(f" - Train: {X_train.shape} (si usa per imparare)")
    print(f" - Val:   {X_val.shape}   (si usa per EarlyStopping)")
    print(f" - Test:  {X_test.shape}  (si usa SOLO alla fine)")

    # Creazione del modello LSTM
    print("Creazione modello LSTM...")
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(MAX_FRAMES, X.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.4),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=8, min_delta=0.00, restore_best_weights=True)

    print("Avvio dell'addestramento...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint, earlystop], verbose=1)

    # --- VALUTAZIONE FINALE ---
    print("\n--- Valutazione Finale sul Test Set ---")
    # Carichiamo i pesi migliori salvati dal Checkpoint per essere sicuri
    model.load_weights(MODEL_PATH) 
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Matrice di confusione sul Test Set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
