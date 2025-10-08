# Rilevamento Violenza in Tempo Reale con MediaPipe e PyTorch

Sistema di preprocessing per il rilevamento di azioni violente (pugnalate, aggressioni) utilizzando MediaPipe per l'estrazione delle pose e PyTorch per il machine learning.

## 📋 Requisiti

### Dipendenze Python
```bash
pip install mediapipe opencv-python torch numpy tqdm
```

### Versioni consigliate
- Python 3.8+
- PyTorch 2.0+
- MediaPipe 0.10+
- OpenCV 4.8+

## 📁 Struttura Dataset

Il tuo dataset deve essere organizzato nel seguente modo:

```
video-dataset/
├── violent/
│   ├── cam1/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── cam2/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── non-violent/
    ├── cam1/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    └── cam2/
        ├── video1.mp4
        ├── video2.mp4
        └── ...
```

### Formati video supportati
- `.mp4`
- `.avi`
- `.mov`

## 🚀 Utilizzo

### 1. Preprocessing dei Video

Esegui lo script per elaborare tutti i video:

```python
python preprocess_videos.py
```

Questo processo:
1. Legge tutti i video dalle cartelle `violent` e `non-violent`
2. Estrae i landmark della posa con MediaPipe (33 punti del corpo)
3. Calcola velocità e accelerazione dei movimenti
4. Salva le feature elaborate in `processed_data/`

### 2. Parametri Configurabili

```python
features, labels, metadata = preprocess_dataset(
    dataset_path='video-dataset',      # Percorso al tuo dataset
    output_path='processed_data',      # Dove salvare i dati elaborati
    sequence_length=30,                # Numero di frame per sequenza
    frame_skip=1,                      # Elabora ogni N frame (1=tutti)
    max_frames_per_video=None          # Limite frame per video (None=tutti)
)
```

#### Parametri Importanti:

- **`sequence_length`**: Numero di frame per campione (30 frame ≈ 1 secondo a 30fps)
- **`frame_skip`**: Velocizza l'elaborazione saltando frame
  - `1` = elabora tutti i frame
  - `2` = elabora un frame sì e uno no
  - `3` = elabora ogni terzo frame
- **`batch_size`**: Numero di video processati insieme durante il training
  - `8-16`: Consigliato per GPU con poca memoria
  - `32-64`: Per GPU più potenti

### 3. Creare DataLoader per il Training

```python
train_loader, val_loader = create_dataloaders(
    features_path='processed_data/features.pkl',
    labels_path='processed_data/labels.pkl',
    sequence_length=30,
    batch_size=16,
    train_split=0.8,      # 80% training, 20% validation
    shuffle=True
)
```

## 📊 Output del Preprocessing

### File Generati:

1. **`features.pkl`**: Feature estratte da tutti i video
   - Dimensione per frame: 396 valori
   - Composizione: landmark (132) + velocità (132) + accelerazione (132)

2. **`labels.pkl`**: Etichette corrispondenti
   - `1` = violento
   - `0` = non violento

3. **`metadata.json`**: Informazioni sui video
   ```json
   {
     "video_path": "percorso/video.mp4",
     "category": "violent",
     "camera": "cam1",
     "num_frames": 120
   }
   ```

4. **`stats.json`**: Statistiche del dataset
   ```json
   {
     "total_videos": 200,
     "violent_videos": 100,
     "non_violent_videos": 100,
     "feature_dim": 396,
     "avg_frames_per_video": 85.5
   }
   ```

## 🧠 Feature Estratte

Per ogni frame, MediaPipe estrae:

### 1. Landmark della Posa (132 valori)
- 33 punti del corpo × 4 valori (x, y, z, visibilità)
- Punti chiave: spalle, gomiti, polsi, anche, ginocchia, caviglie, ecc.

### 2. Velocità (132 valori)
- Cambio di posizione tra frame consecutivi
- Utile per rilevare movimenti rapidi (pugni, pugnalate)

### 3. Accelerazione (132 valori)
- Cambio di velocità tra frame
- Utile per rilevare movimenti improvvisi

**Totale: 396 feature per frame**


## 🎯 Prossimi Passi

Dopo il preprocessing, puoi:

1. **Allenare un modello LSTM** per la classificazione sequenziale
2. **Usare un Transformer** per catturare relazioni a lungo termine
3. **Implementare un CNN+LSTM** per feature spaziali e temporali
4. **Testare in tempo reale** con webcam o video stream

## 🐛 Risoluzione Problemi

### Errore: "No features extracted"
- Verifica che i video siano leggibili con OpenCV
- Controlla che le persone siano visibili nei video
- Prova ad abbassare `min_detection_confidence`

### Errore di Memoria (Out of Memory)
- Riduci `batch_size`
- Aumenta `frame_skip`
- Riduci `sequence_length`

### Video Non Trovati
- Verifica la struttura delle cartelle
- Controlla i nomi: `violent`, `non-violent`, `cam1`, `cam2`
- Verifica le estensioni dei file (`.mp4`, `.avi`, `.mov`)

## 📝 Note

- Il preprocessing può richiedere tempo (diversi minuti/ore a seconda del dataset)
- MediaPipe funziona meglio con persone ben inquadrate e illuminate
- Le feature di velocità e accelerazione sono cruciali per rilevare movimenti violenti
- I dati da cam1 e cam2 vengono trattati come campioni separati (data augmentation naturale)

## 📧 Supporto

Per problemi o domande, verifica:
1. Che tutti i pacchetti siano installati correttamente
2. Che la struttura del dataset sia corretta
3. Che i video siano in formato supportato e riproducibili

---