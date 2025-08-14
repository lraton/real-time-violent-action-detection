# Pre-elaborazione dei Video (`preprocess_video.py`)

Questo script Python esegue **OpenPose** su tutti i file video (`.mp4`) trovati in una specifica cartella e salva i keypoint del corpo rilevati in file **JSON**.

---

## Configurazione e Dipendenze

- **OpenPose**  
  Devi avere OpenPose installato e specificare il percorso dell’eseguibile nella variabile `OPENPOSE_EXECUTABLE`.  
  Lo script è configurato per la versione *portable* di OpenPose per Windows.

- **Struttura delle cartelle**  
  Assicurati che i tuoi file video siano organizzati all’interno della cartella specificata da `VIDEO_ROOT_FOLDER`.  
  Lo script manterrà la stessa struttura di cartelle nella directory di output.

- **Librerie Python**  
  - `os`  
  - `subprocess`  
  - `tqdm` (installabile con `pip install tqdm`)

---

## ⚙️ Come Funziona

1. **Impostazioni**  
   Definisce i percorsi per:
   - l’eseguibile di OpenPose  
   - la cartella dei video (`video_dataset`)  
   - la cartella di output per i JSON (`json_output`)

2. **Scansione dei video**  
   Percorre ricorsivamente la cartella `video_dataset` per trovare tutti i file `.mp4`.

3. **Esecuzione di OpenPose**  
   Per ogni video trovato, esegue OpenPose come processo esterno.  
   - Flag usati:  
     - `--display 0` → disabilita la visualizzazione  
     - `--render_pose 0` → disabilita l’output video (più veloce)  
   - I keypoint vengono salvati in una cartella di output con lo stesso nome del video, contenente un file JSON per ogni frame.

4. **Gestione errori**  
   Controlla eventuali errori nell’esecuzione di OpenPose e salta i video già elaborati.

---
