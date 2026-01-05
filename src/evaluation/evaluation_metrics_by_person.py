import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score)


# ----- FUNZIONE FLICKER RATE -----
def calculate_flicker_rate(df):
    """
    Calcola quanto spesso la predizione cambia stato per la stessa persona.
    Più basso è, più stabile è il modello.
    """
    flicker_scores = []

    # Raggruppa per video e persona per analizzare la sequenza temporale
    grouped = df.groupby(["video_id", "person_id"])

    for (video_id, person_id), group in grouped:
        # Ordina per frame per garantire la sequenza temporale
        group = group.sort_values("frame_id")

        preds = group["pred_class"].values

        # Servono almeno 2 frame per calcolare il cambio
        if len(preds) < 2:
            continue

        # Calcola i cambiamenti di stato (es. da 0 a 1, da 1 a 0)
        changes = np.sum(preds[1:] != preds[:-1])

        # Normalizza sul numero di frame
        score = changes / (len(preds) - 1)
        flicker_scores.append(score)

    return np.mean(flicker_scores) if flicker_scores else 0.0


def main():
    # ----- Caricamento e Preparazione Dati -----
    filename = "evaluation_results/predictions_v8_v2.csv"
    try:
        df = pd.read_csv(filename)
        print(f"Caricati {len(df)} frame totali da {filename}")
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return

    # Pulizia e casting
    df['pred_class'] = pd.to_numeric(df['pred_class'], errors='coerce').fillna(0).astype(int)
    df['true_class'] = pd.to_numeric(df['true_class'], errors='coerce').fillna(0).astype(int)

    # Creazione "Score Unificato" (serve per l'AUC)
    # Se ha un coltello (1.0) vince su tutto, altrimenti usa violence_score
    df["unified_score"] = df.apply(lambda row: 1.0 if row.get("has_knife", 0) == 1 else row.get("violence_score", 0), axis=1)

    # ----- Calcolo Flicker Rate (Prima dell'aggregazione) -----
    avg_flicker = calculate_flicker_rate(df)

    # ----- Logica di Aggregazione (PER PERSONA) -----
    print("Elaborazione raggruppamento per persona...")

    grouped = df.groupby(['video_id', 'person_id'])

    y_true_people = []
    y_pred_people = []
    y_score_people = []  # Serve per l'AUC

    for name, group in grouped:
        # A. GROUND TRUTH (Realtà)
        # Se la persona ha fatto anche solo 1 frame di accoltellamento (2), è etichettata come 2.
        person_true_class = group['true_class'].max()

        # B. SCORE (Per AUC)
        # Prendiamo il picco massimo di pericolosità raggiunto dalla persona
        person_max_score = group['unified_score'].max()

        # C. PREDIZIONE (Logica Soglia 1%)
        violent_frames = group[group['pred_class'] > 0]  # Frame predetti 1 o 2
        percentage_violent = len(violent_frames) / len(group)

        if percentage_violent > 0.01:  # Se > 1% dei frame sono violenti
            # Prendiamo la classe violenta più grave predetta
            person_pred_class = group['pred_class'].max()
        else:
            # Sotto soglia, consideriamo falso allarme -> 0 (Calmo)
            person_pred_class = 0

        y_true_people.append(person_true_class)
        y_pred_people.append(person_pred_class)
        y_score_people.append(person_max_score)

    # Convertiamo in array numpy per comodità
    y_true_people = np.array(y_true_people)
    y_pred_people = np.array(y_pred_people)
    y_score_people = np.array(y_score_people)

    print(f"Analisi completata su {len(y_true_people)} persone uniche.")

    # ----- Report Testuale -----
    print("\n" + "=" * 40)
    print("=== REPORT PER SOGGETTO (PERSON-LEVEL) ===")
    print("=" * 40)

    target_names_map = {0: "Non violento", 1: "Aggressione", 2: "Accoltellamento"}
    unique_lbls = sorted(np.unique(np.concatenate((y_true_people, y_pred_people))))
    target_names = [target_names_map[i] for i in unique_lbls]

    print(classification_report(y_true_people, y_pred_people, labels=unique_lbls, target_names=target_names, zero_division=0))

    # ----- Matrice di Confusione (Grafica) -----
    cm = confusion_matrix(y_true_people, y_pred_people)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (Livello Persona)')
    plt.ylabel('Reale (Ground Truth)')
    plt.xlabel('Predetto dal Sistema')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix_person_v8_byperson.png', bbox_inches='tight')
    print("Grafico salvato in: evaluation_results/confusion_matrix_person_v8_byperson.png")

    # ----- Metriche Avanzate (AUC - Safe vs Danger) -----
    print("\n" + "=" * 40)
    print("=== METRICHE GLOBALI PERSONA (Safe vs Danger) ===")

    # Binarizziamo: 0 = Safe, 1/2 = Danger
    y_true_bin = (y_true_people > 0).astype(int)

    if len(np.unique(y_true_bin)) > 1:
        roc_auc = roc_auc_score(y_true_bin, y_score_people)
        pr_auc = average_precision_score(y_true_bin, y_score_people)

        print(f"ROC-AUC Score : {roc_auc:.4f} (Discriminazione generale)")
        print(f"PR-AUC Score  : {pr_auc:.4f} (Affidabilità sui casi positivi)")

        if pr_auc > 0.8:
            print(">> Risultato: ECCELLENTE.")
        elif pr_auc > 0.5:
            print(">> Risultato: BUONO.")
        else:
            print(">> Risultato: SCARSO.")
    else:
        print("Impossibile calcolare AUC (manca una delle due classi).")

    # ----- Stampa Flicker Rate -----
    print("\n" + "=" * 40)
    print(f"STABILITÀ SISTEMA (Flicker Rate): {avg_flicker:.4f}")
    if avg_flicker < 0.02:
        print(">> ECCELLENTE: Il tracking è stabile.")
    elif avg_flicker < 0.05:
        print(">> BUONO: Qualche sfarfallio.")
    else:
        print(">> ATTENZIONE: Il sistema è instabile.")
    print("=" * 40)

    # Mostra grafico
    plt.show()


if __name__ == "__main__":
    main()
