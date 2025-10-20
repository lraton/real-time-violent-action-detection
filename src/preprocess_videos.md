# 🧠 YOLO Pose Keypoint Extractor

Questo progetto utilizza **Ultralytics YOLO** per estrarre e normalizzare i keypoint umani da video, con l’obiettivo di creare dataset per modelli di **Action Recognition** (ad esempio per distinguere azioni violente e non violente).

---

## 📦 Requisiti

- Ultralytics YOLO  
- OpenCV  
- NumPy  
- Natsort  

Le dipendenze possono essere installate tramite `pip`.

---

## ⚙️ Descrizione del processo

### Caricamento del modello
Viene caricato un modello YOLO pre-addestrato per la stima della posa (formato COCO) che consente di rilevare i keypoint umani in ogni frame dei video.

### Analisi dei video
Lo script scansiona una directory contenente video e per ciascuno di essi avvia un processo di tracking multi-persona. Ogni individuo rilevato mantiene un ID coerente per tutto il video.

### Normalizzazione dei keypoint
Per rendere i dati indipendenti dalla posizione nel frame, i keypoint vengono normalizzati rispetto al centro del torace, calcolato come punto medio tra la spalla sinistra e la destra.

### Salvataggio dei dati
Per ogni persona tracciata, i keypoint raccolti durante tutti i frame vengono salvati in file `.npz`.  
Ogni file contiene i keypoint normalizzati e un’etichetta che identifica il tipo di azione osservata.

### Visualizzazione
È possibile abilitare una visualizzazione in tempo reale del video con box e keypoint annotati.  
Durante la riproduzione, è possibile interrompere l’esecuzione con il tasto **Q**.

---

## 🚀 Esecuzione
Lo script può essere eseguito direttamente da terminale. Durante l’elaborazione, mostra il nome del video in analisi e il numero di persone tracciate.

---

## Formato npy
- `data`: array NumPy con i keypoint normalizzati, di forma `(num_frames, num_keypoints, 3)` dove il terzo asse contiene `(x, y, confidence)`  
- `label`: intero che indica il tipo di azione osservata (1 = violento, 0 = non violento)  