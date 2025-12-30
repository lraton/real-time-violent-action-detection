import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

# --- CONFIGURAZIONE ---
DATA_PATH = "../datasets/lstm_dataset/"
MODEL_PATH = "../models/lstm_violence_detector_v8.keras"
MAX_FRAMES = 150
BATCH_SIZE = 32  
EPOCHS = 100
MASK_VALUE = -1.0  # Usiamo -1 per il padding


def main():
    X, y = [], []

    print("Caricamento file .npz...")
    # Caricamento dati
    for file in glob.glob(os.path.join(DATA_PATH, "*.npz")):
        data = np.load(file)
        X.append(data["data"])
        y.append(int(data["label"]))

    print(f"Totale sequenze: {len(X)}")

    # 1. Normalizzazione reale e Padding
    X, y = preprocess_data(X, y)

    # 2. Addestramento
    create_and_train_model(X, y)


def preprocess_data(X, y):
    print("Preprocessing e Normalizzazione...")

    # STEP A: Normalizzazione (Assumendo che i keypoints siano grezzi)
    # Sarebbe meglio normalizzare durante la creazione del dataset,
    # ma se non puoi, assicurati che i dati qui siano tra 0 e 1.
    # Se sono già normalizzati nel .npz, salta questo commento.

    # STEP B: Padding con valore speciale
    # Usiamo 'pre' padding: aiuta la LSTM a ricordare meglio la fine dell'azione
    X = pad_sequences(X, maxlen=MAX_FRAMES, dtype='float32', padding='post', truncating='post', value=MASK_VALUE)

    # Reshape
    X = X.reshape(X.shape[0], X.shape[1], -1)
    y = np.array(y, dtype=np.float32)

    print(f"Forma finale X: {X.shape}, y: {y.shape}")
    return X, y


def create_and_train_model(X, y):
    # Split Stratificato
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Architettura Migliorata
    print("Creazione modello Bidirectional LSTM...")
    model = Sequential([
        # Maschera il valore di padding (-1.0) invece di 0.0
        Masking(mask_value=MASK_VALUE, input_shape=(MAX_FRAMES, X.shape[2])),

        # Bidirectional permette di capire il contesto temporale in entrambe le direzioni
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),

        # BatchNormalization aiuta a stabilizzare l'apprendimento
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks potenziati
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Riduce il Learning Rate se l'addestramento stalla
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

    print("Avvio training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=None,  # Se i dati sono 50/50 non serve, altrimenti calcolali
        callbacks=[checkpoint, earlystop, reduce_lr],
        verbose=1)

    # Valutazione
    print("\n--- Report sul Test Set (Binario) ---")
    model.load_weights(MODEL_PATH)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    print(classification_report(y_test, y_pred, target_names=['Non Violento', 'Violento']))


if __name__ == '__main__':
    main()
