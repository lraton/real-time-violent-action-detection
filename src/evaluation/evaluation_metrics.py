import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score)


def main():
    # ----- Caricamento Dati -----
    filename = "predictions.csv"  # O "evaluation/predictions.csv"
    try:
        df = pd.read_csv(filename)
        print(f"Caricati {len(df)} record da {filename}")
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return

    # Controlliamo che ci siano le colonne necessarie
    required_cols = ["pred_class", "true_class", "violence_score", "has_knife"]
    if not all(col in df.columns for col in required_cols):
        print(f"Errore: Mancano alcune colonne nel CSV. Assicurati che ci siano: {required_cols}")
        return

    # Filtriamo eventuali righe dove true_class non è valido (es. test live senza etichetta)
    df = df.dropna(subset=["true_class"])
    # Assicuriamoci che siano interi
    df["true_class"] = df["true_class"].astype(int)
    df["pred_class"] = df["pred_class"].astype(int)

    y_true = df["true_class"]
    y_pred = df["pred_class"]

    class_names = ["Non violento", "Aggressione", "Accoltellamento"]

    # ----- Creazione "Score Unificato" per le curve AUC -----

    # Il problema: "violence_score" misura solo la violenza fisica (LSTM).
    # Se c'è un coltello (has_knife=1), la situazione è grave (1.0) anche se l'LSTM dice 0.1.
    # Creiamo uno score che combina le due cose per valutare la "Pericolosità totale".

    df["unified_score"] = df.apply(lambda row: 1.0 if row["has_knife"] == 1 else row["violence_score"], axis=1)
    y_score = df["unified_score"]

    # ----- Report Testuale (Precision, Recall, F1) -----

    print("\n" + "=" * 40)
    print("=== CLASSIFICATION REPORT ===")
    labels = sorted(np.unique(y_true))

    name_map = {0: "Non violento", 1: "Aggressione", 2: "Accoltellamento"}

    class_names = [name_map[l] for l in labels]

    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=3, zero_division=0))

    # ----- Matrice di Confusione -----

    cm = confusion_matrix(y_true, y_pred)

    # Plotting 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Reale (Ground Truth)')
    plt.xlabel('Predetto dal Sistema')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight')

    # ----- Metriche Avanzate (AUC) - Binario: Safe vs Danger -----

    # Raggruppiamo Aggressione(1) e Accoltellamento(2) in "Pericolo"(1)
    y_true_bin = (y_true > 0).astype(int)

    # Calcoliamo solo se nel file ci sono sia esempi positivi che negativi
    if len(np.unique(y_true_bin)) > 1:
        roc_auc = roc_auc_score(y_true_bin, y_score)
        pr_auc = average_precision_score(y_true_bin, y_score)

        print("\n" + "=" * 40)
        print("=== METRICHE GLOBALI (Safe vs Danger) ===")
        print(f"ROC-AUC Score : {roc_auc:.4f} (Capacità di discriminazione generale)")
        print(f"PR-AUC Score  : {pr_auc:.4f} (Cruciale per dataset sbilanciati)")

        # Interpretazione veloce
        if pr_auc > 0.8:
            print(">> Risultato: ECCELLENTE. Il sistema è molto affidabile.")
        elif pr_auc > 0.5:
            print(">> Risultato: BUONO. Utile, ma con margini di miglioramento.")
        else:
            print(">> Risultato: SCARSO. Molti falsi allarmi o eventi persi.")
    else:
        print("\nAttenzione: Impossibile calcolare AUC. Il CSV contiene solo una classe (tutti violenti o tutti calmi).")

    # ----- FLICKER RATE -----
    def calculate_flicker_rate(df):
        flicker_scores = []

        # Raggruppa per video (non vogliamo calcolare il cambio tra la fine del video A e l'inizio del video B)
        grouped = df.groupby("video_id")

        for video_id, group in grouped:
            # Ordina per frame per essere sicuri della sequenza temporale
            group = group.sort_values("frame_id")

            preds = group["pred_class"].values

            if len(preds) < 2:
                continue

            # Calcola quanti cambi di stato ci sono (es. da 0 a 1, da 1 a 0, da 1 a 2...)
            # preds[:-1] prende tutti tranne l'ultimo
            # preds[1:] prende tutti tranne il primo
            # Confrontandoli, vediamo se lo stato al tempo T è diverso da T-1
            changes = np.sum(preds[1:] != preds[:-1])

            # Formula: Cambiamenti / (Totale Frame - 1)
            score = changes / (len(preds) - 1)
            flicker_scores.append(score)

            # Opzionale: stampa i video peggiori
            if score > 0.1:
                print(f"Video instabile: {video_id} (Flicker: {score:.3f})")

        return np.mean(flicker_scores) if flicker_scores else 0.0

    # Calcola
    avg_flicker = calculate_flicker_rate(df)

    print("\n" + "=" * 40)
    print(f"AVERAGE FLICKER RATE: {avg_flicker:.4f}")
    print("=" * 40)

    # ----- INTERPRETAZIONE -----
    if avg_flicker < 0.02:
        print(">> ECCELLENTE: Il sistema è molto stabile.")
    elif avg_flicker < 0.05:
        print(">> BUONO: Qualche sfarfallio, ma accettabile.")
    else:
        print(">> ATTENZIONE: Il sistema è instabile (cambia idea troppo spesso).")
    
    plt.show()

if __name__ == "__main__":
    main()
