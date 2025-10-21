# Trainind modello lstm per rilevare violenza

Progetto per rilevare comportamenti violenti nei video utilizzando sequenze di keypoint e un modello LSTM.

---

- **Cartella dati (`DATA_PATH`)**: contiene file `.npz` con:
  - `data`: sequenze di keypoint
  - `label`: 0 (non violento) o 1 (violento)
- **Percorso modello (`MODEL_PATH`)**: dove viene salvato il modello addestrato.
- **Parametri principali**:
  - `MAX_FRAMES`: numero massimo di frame per sequenza
  - `BATCH_SIZE`: dimensione del batch
  - `EPOCHS`: numero massimo di epoche

---

## Flusso principale

### 1. Caricamento dati
- Lettura di tutti i file `.npz` nella cartella dati.
- Estrazione di sequenze (`X`) e label (`y`).

### 2. Normalizzazione delle sequenze
- Padding o troncamento delle sequenze a `MAX_FRAMES`.
- Conversione in array `float32`.
- Reshape per adattarsi all’input LSTM.
- Conversione delle label in array `float32`.

### 3. Creazione del modello LSTM
- Suddivisione in **training** e **validation set** (80%/20%).
- Architettura del modello:
  - Mascheramento dei valori nulli (padding)
  - LSTM con 128 unità (`return_sequences=True`)
  - Dropout (0.3)
  - LSTM con 64 unità
  - Dense intermedio con 32 neuroni, attivazione `ReLU`
  - Dense finale con 1 neurone, attivazione `sigmoid`
- Compilazione: optimizer `Adam`, loss `binary_crossentropy`
- Callbacks:
  - **ModelCheckpoint**: salva il miglior modello
  - **EarlyStopping**: interrompe se la loss non migliora per 8 epoche

### 4. Addestramento
- Fit sul training set con validazione sul validation set
- Salvataggio del modello migliore in `MODEL_PATH`

---