# Addestramento del Modello (`train_model.py`)

Questo script utilizza i file **JSON** generati dal primo script per addestrare un modello di deep learning **CNN+LSTM** per la classificazione di azioni.

---

## Architettura del Modello

L’architettura è progettata per analizzare sequenze temporali di dati di **pose**:

1. **Input**  
   Sequenze di keypoint *(coordinate x, y)* estratte da ciascun frame.

2. **CNN (Convolutional Neural Network)**  
   Utilizza un livello lineare (`nn.Linear`) per estrarre feature significative da ogni frame, trasformando le coordinate grezze in un vettore di feature più denso.

3. **LSTM (Long Short-Term Memory)**  
   Un layer LSTM elabora la sequenza di feature, catturando dipendenze temporali e contesto dell’azione.

4. **Output**  
   L’output dell’ultimo step dell’LSTM viene passato a un layer completamente connesso (`nn.Linear`) per classificare l’azione in una delle classi predefinite:
   - `stab`
   - `other_violent`
   - `non-violent`

---

## Come Funziona

1. **Dataset Personalizzato**  
   La classe `PoseDataset` carica i file JSON, estrae i keypoint e applica *padding* o *troncamento* per garantire la stessa lunghezza di sequenza (`max_seq_len`).

2. **Preparazione dei Dati**  
   - Carica i percorsi dei file JSON dalle cartelle `violent` e `non-violent`.  
   - Assegna le etichette di classe.  
   - Converte le etichette da stringhe a numeri interi.

3. **Suddivisione**  
   Divide il dataset in:
   - **Training set**
   - **Validation set**

4. **Addestramento**  
   - Durata: 10 epoche  
   - Ottimizzatore: `Adam`  
   - Funzione di perdita: `CrossEntropyLoss`

5. **Valutazione**  
   Alla fine di ogni epoca, il modello viene testato sul set di validazione per monitorare le performance.

6. **Salvataggio**  
   Il modello finale viene salvato come:
 
---