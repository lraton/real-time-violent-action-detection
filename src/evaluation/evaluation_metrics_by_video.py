import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score)

INPUT_CSV = "evaluation_results/predictions_v8_v2.csv"
OUTPUT_IMAGE = "evaluation_results/confusion_matrix_v8_byvideo.png"
OUTPUT_TXT_FILE = "evaluation_results/report_video_metrics_v8.txt"

def print_dataset_stats(df):
    """
    Stampa le statistiche dei VIDEO e i NOMI DEI FILE divisi per categoria (Ground Truth).
    Logica: 
    - Priorità: Accoltellamento (2) > Aggressione (1) > Safe (0).
    """
    print("\n" + "="*60)
    print("=== STATISTICHE E LISTA FILE (GROUND TRUTH) ===")
    print("="*60)

    # 1. Determina la classe reale del video (prendendo il valore massimo)
    video_labels = df.groupby('video_id')['true_class'].max()

    # 2. Definisci le categorie
    categories = {
        0: "NON VIOLENTI (Safe)",
        1: "AGGRESSIONE (Rissa/Pugni)",
        2: "ACCOLTELLAMENTO (Arma)"
    }

    # 3. Itera per stampare totali e nomi file
    total_videos = len(video_labels)
    
    for class_id, label in categories.items():
        # Filtra i video che appartengono a questa classe
        files = video_labels[video_labels == class_id].index.tolist()
        count = len(files)
        
        print(f"\n>>> {label} | Totale: {count} ({count/total_videos*100:.1f}%)")
        print("-" * 40)
        
        if count > 0:
            # Ordina alfabeticamente per lettura più facile
            for f in sorted(files):
                print(f"  • {f}")
        else:
            print("  (Nessun video trovato)")

    print("\n" + "="*60 + "\n")

# ----- FRAME CONSECUTIVI -----
def has_consecutive_violence(group, threshold=5):
    # Ordiniamo per frame per sicurezza
    group = group.sort_values("frame_id")
    
    # Creiamo una serie booleana: True se è violento (1 o 2), False se è 0
    is_violent = group["pred_class"] > 0
    
    # Calcolo delle sequenze consecutive
    consecutive_groups = is_violent.groupby((is_violent != is_violent.shift()).cumsum())
    
    # Calcoliamo la dimensione di ogni blocco, ma solo se il blocco è violento
    max_consecutive = 0
    for _, grp in consecutive_groups:
        if grp.iloc[0]: # Se è un blocco di 'True' (Violenza)
            if len(grp) > max_consecutive:
                max_consecutive = len(grp)
    
    return max_consecutive >= threshold

# ----- FUNZIONE FLICKER RATE -----
def calculate_flicker_rate(df):
    flicker_scores = []
    if 'person_id' in df.columns:
        grouped = df.groupby(["video_id", "person_id"])
    else:
        grouped = df.groupby(["video_id"])

    for _, group in grouped:
        group = group.sort_values("frame_id")
        preds = group["pred_class"].values
        if len(preds) < 2: continue
        changes = np.sum(preds[1:] != preds[:-1])
        score = changes / (len(preds) - 1)
        flicker_scores.append(score)
    return np.mean(flicker_scores) if flicker_scores else 0.0

def main():
    filename = INPUT_CSV
    
    # --- CONFIGURAZIONE SOGLIA ---
    CONSECUTIVE_THRESHOLD = 30

    try:
        df = pd.read_csv(filename)
        print(f"Caricati {len(df)} frame totali da {filename}")
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return
    print_dataset_stats(df)

    # Pulizia e casting
    df['pred_class'] = pd.to_numeric(df['pred_class'], errors='coerce').fillna(0).astype(int)
    df['true_class'] = pd.to_numeric(df['true_class'], errors='coerce').fillna(0).astype(int)

    # Score unificato per AUC
    df["unified_score"] = df.apply(lambda row: 1.0 if row.get("has_knife", 0) == 1 else row.get("violence_score", 0), axis=1)

    # Calcolo Flicker
    avg_flicker = calculate_flicker_rate(df)

    # ----- ANALISI PER VIDEO -----
    print(f"Analisi Video in corso (Soglia frame consecutivi: {CONSECUTIVE_THRESHOLD})...")

    grouped = df.groupby(['video_id'])

    y_true_video = []
    y_pred_video = []
    y_score_video = [] 

    for video_id, video_group in grouped:
        
        # Se c'è violenza vera in qualsiasi punto, il video è violento
        if video_group['true_class'].max() > 0:
            video_true_class = 1
        else:
            video_true_class = 0

        # --- PREDIZIONE ---
        video_is_predicted_violent = False
        
        # Raggruppiamo per persone all'interno di questo video
        people_in_video = video_group.groupby('person_id')
        
        # Controlliamo per ogni persona quanti frame consecutivi di violenza ci sono
        for person_id, person_data in people_in_video:
            # Chiamiamo la funzione che conta i frame consecutivi
            if has_consecutive_violence(person_data, threshold=CONSECUTIVE_THRESHOLD):
                video_is_predicted_violent = True
                break # Basta che una persona sia violenta per marcare il video
        
        video_pred_class = 1 if video_is_predicted_violent else 0

        # --- SCORE (Per AUC) ---
        video_max_score = video_group['unified_score'].max()

        y_true_video.append(video_true_class)
        y_pred_video.append(video_pred_class)
        y_score_video.append(video_max_score)

    # Casting array
    y_true_video = np.array(y_true_video)
    y_pred_video = np.array(y_pred_video)
    y_score_video = np.array(y_score_video)

    print(f"Analisi completata su {len(y_true_video)} video unici.")

    # ----- SALVATAGGIO REPORT SU FILE -----
    
    with open(OUTPUT_TXT_FILE, "w") as f:
        
        # Funzione helper per scrivere sia su console che su file
        def log(text):
            print(text)           # Stampa a video
            f.write(text + "\n")  # Scrive nel file

        # 1. REPORT CLASSIFICAZIONE
        log("\n" + "=" * 56)
        log(f"=== REPORT PER VIDEO (Soglia Consecutiva: {CONSECUTIVE_THRESHOLD} frames) ===")
        log("=" * 56)

        target_names = ["Non Violento", "Violento"]
        
        # Generiamo il report come stringa
        class_report = classification_report(y_true_video, y_pred_video, target_names=target_names, zero_division=0)
        log(class_report)

        # 2. METRICHE GLOBALI (AUC)
        if len(np.unique(y_true_video)) > 1:
            log("\n" + "=" * 56)
            log("=== METRICHE GLOBALI ===")
            
            roc = roc_auc_score(y_true_video, y_score_video)
            pr = average_precision_score(y_true_video, y_score_video)
            
            log(f"ROC-AUC Score : {roc:.4f}")
            log(f"PR-AUC Score  : {pr:.4f}")

        # 3. FLICKER RATE
        log("\n" + "=" * 56)
        log(f"FLICKER RATE: {avg_flicker:.4f}")
        log("=" * 56)

    print(f"\n[INFO] Report testuale salvato in: {OUTPUT_TXT_FILE}")

    # ----- SALVATAGGIO GRAFICO MATRICE -----
    # Confusion Matrix (Questa parte rimane uguale per l'immagine)
    cm = confusion_matrix(y_true_video, y_pred_video)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (Video Level)\nThreshold: {CONSECUTIVE_THRESHOLD} consecutive frames')
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.tight_layout()
    
    # Assicurati di avere definito OUTPUT_IMAGE prima, ad esempio:
    OUTPUT_IMAGE = "evaluation_results/confusion_matrix_video.png"
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
    print(f"[INFO] Grafico salvato in: {OUTPUT_IMAGE}")

    plt.show()

if __name__ == "__main__":
    main()