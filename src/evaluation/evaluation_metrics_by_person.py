import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score)

INPUT_CSV = "evaluation_results/predictions_v8_v2.csv"
OUTPUT_IMAGE = "evaluation_results/confusion_matrix_v8_v3_byperson.png"

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


# ----- FLICKER RATE -----
def calculate_flicker_rate(df):
    """
    Calcola la stabilità delle predizioni.
    """
    flicker_scores = []
    grouped = df.groupby(["video_id", "person_id"])

    for (video_id, person_id), group in grouped:
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
    CONSECUTIVE_THRESHOLD = 2  # Es. 8 frame consecutivi (~0.25s a 30fps)

    try:
        df = pd.read_csv(filename)
        print(f"Caricati {len(df)} frame totali da {filename}")
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return

    # Pulizia
    df['pred_class'] = pd.to_numeric(df['pred_class'], errors='coerce').fillna(0).astype(int)
    df['true_class'] = pd.to_numeric(df['true_class'], errors='coerce').fillna(0).astype(int)

    # Score unificato
    df["unified_score"] = df.apply(lambda row: 1.0 if row.get("has_knife", 0) == 1 else row.get("violence_score", 0), axis=1)

    # Calcolo Flicker
    avg_flicker = calculate_flicker_rate(df)

    # ----- Logica di Aggregazione -----
    print(f"Raggruppamento per Persona (Soglia Consecutiva: {CONSECUTIVE_THRESHOLD})...")

    grouped = df.groupby(['video_id', 'person_id'])

    y_true_people = []
    y_pred_people = []
    y_score_people = [] 

    for name, group in grouped:

        # Se c'è violenza vera in qualsiasi punto, il video è violento
        person_true_class = group['true_class'].max()

        # SCORE (Per AUC)
        person_max_score = group['unified_score'].max()

        # --- PREDIZIONE ---
        # Controlliamo se questa specifica persona ha una sequenza violenta
        if (group['pred_class'] == 2).any():
            person_pred_class = 2
        elif has_consecutive_violence(group, threshold=CONSECUTIVE_THRESHOLD):
            violent_preds = group[group['pred_class'] > 0]['pred_class']
            if not violent_preds.empty:
                person_pred_class = violent_preds.max()
            else:
                person_pred_class = 1 # Fallback (improbabile se la funzione sopra è True)
        else:
            # Non ha abbastanza frame consecutivi -> consideriamo falso allarme
            person_pred_class = 0 

        y_true_people.append(person_true_class)
        y_pred_people.append(person_pred_class)
        y_score_people.append(person_max_score)

    # Convertiamo in array
    y_true_people = np.array(y_true_people)
    y_pred_people = np.array(y_pred_people)
    y_score_people = np.array(y_score_people)

    print(f"Analisi completata su {len(y_true_people)} persone uniche.")

    # ----- Report Testuale -----
    print("\n" + "=" * 56)
    print("=== REPORT PER SOGGETTO (PERSON-LEVEL) ===")
    print(f"Soglia validazione: {CONSECUTIVE_THRESHOLD} frame consecutivi")
    print("=" * 56)

    target_names_map = {0: "Non violento", 1: "Aggressione", 2: "Accoltellamento"}
    unique_lbls = sorted(np.unique(np.concatenate((y_true_people, y_pred_people))))
    target_names = [target_names_map[i] for i in unique_lbls]

    print(classification_report(y_true_people, y_pred_people, labels=unique_lbls, target_names=target_names, zero_division=0))

    # ----- Matrice di Confusione -----
    cm = confusion_matrix(y_true_people, y_pred_people)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (Persona)\nSoglia: {CONSECUTIVE_THRESHOLD} frame cons.')
    plt.ylabel('Reale (Ground Truth)')
    plt.xlabel('Predetto dal Sistema')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
    print(f"Grafico salvato in: {OUTPUT_IMAGE}")

    # ----- Metriche Avanzate (Safe vs Danger) -----
    print("\n" + "=" * 56)
    print("=== METRICHE GLOBALI PERSONA ===")

    y_true_bin = (y_true_people > 0).astype(int)

    if len(np.unique(y_true_bin)) > 1:
        roc_auc = roc_auc_score(y_true_bin, y_score_people)
        pr_auc = average_precision_score(y_true_bin, y_score_people)

        print(f"ROC-AUC Score : {roc_auc:.4f}")
        print(f"PR-AUC Score  : {pr_auc:.4f}")

        if pr_auc > 0.8: print(">> Risultato: ECCELLENTE.")
        elif pr_auc > 0.5: print(">> Risultato: BUONO.")
        else: print(">> Risultato: SCARSO.")
    else:
        print("Impossibile calcolare AUC.")

    # ----- 6. Stampa Flicker Rate -----
    print("\n" + "=" * 56)
    print(f"STABILITÀ SISTEMA (Flicker Rate): {avg_flicker:.4f}")
    print("=" * 56)

    plt.show()

if __name__ == "__main__":
    main()