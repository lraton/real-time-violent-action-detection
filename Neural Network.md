- Estrazione delle Pose: Utilizzi OpenPose per elaborare ogni fotogramma dei tuoi video. OpenPose analizza le immagini e rileva i punti chiave del corpo (spalle, gomiti, polsi, ecc.), generando per ogni persona un'immagine scheletrica o un set di coordinate 2D o 3D. Questi dati rappresentano la posa del corpo in quel preciso istante.

- Preparazione dei Dati: I dati estratti da OpenPose (le sequenze di pose) devono essere preparati per l'addestramento. Ogni sequenza di pose, corrispondente a un'azione, deve essere etichettata con il nome dell'azione che rappresenta (es. "camminare", "saltare", "sedersi").

- Architettura della Rete Neurale: La combinazione di una rete neurale convoluzionale (CNN) e una rete neurale ricorrente a memoria a lungo termine (LSTM) è particolarmente adatta per questo compito.
    - La CNN ha il compito di analizzare ogni singolo frame di pose. Impara a riconoscere le caratteristiche spaziali di una posa, ovvero la configurazione del corpo in un dato momento.

    - La LSTM riceve in input le caratteristiche estratte dalla CNN per ogni frame e impara a riconoscere le dipendenze temporali tra i diversi frame. In pratica, "ricorda" la sequenza di pose e come queste si evolvono nel tempo, che è l'essenza stessa di un'azione.

- Addestramento e Valutazione: Addestri il modello con i tuoi dati etichettati. La rete impara a mappare le sequenze di pose alle rispettive azioni.