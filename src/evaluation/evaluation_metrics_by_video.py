import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score)

INPUT_CSV = "evaluation_results/predictions_v8_v2.csv"
OUTPUT_IMAGE = "confusion_matrix/by_video/confusion_matrix_v8_v2_byvideo_final"
OUTPUT_CURVE = "precision_curve_analysis_v8_v2_final.png"
OUTPUT_TXT_FILE = "text_report/report_video_metrics_v8_v2_final"
OUTPUT_DIR = "evaluation_results"

'''
def print_dataset_stats(df):
    """
    Stampa le statistiche dei VIDEO, i NOMI DEI FILE e il CONTEGGIO PERSONE.
    Logica: 
    - Priorità: Accoltellamento (2) > Aggressione (1) > Safe (0).
    """
    print("\n" + "="*60)
    print("=== STATISTICHE E LISTA FILE (GROUND TRUTH) ===")
    print("="*60)

    # 1. Determina la classe reale del video
    video_labels = df.groupby('video_id')['true_class'].max()

    # 2. CONTEGGIO PERSONE UNICHE
    # Usiamo una combinazione di video_id e person_id per identificare univocamente un soggetto
    unique_subjects = df.groupby(['video_id', 'person_id']).size().reset_index()
    total_unique_persons = len(unique_subjects)

    # 3. Definisci le categorie
    categories = {
        0: "NON VIOLENTI (Safe)",
        1: "AGGRESSIONE (Rissa/Pugni)",
        2: "ACCOLTELLAMENTO (Arma)"
    }

    total_videos = len(video_labels)
    
    # 4. Stampa statistiche generali
    print(f"VIDEO TOTALI ANALIZZATI: {total_videos}")
    print(f"PERSONE TOTALI RILEVATE: {total_unique_persons}")
    if total_videos > 0:
        print(f"MEDIA PERSONE PER VIDEO: {total_unique_persons/total_videos:.2f}")
    print("-" * 60)

    # 5. Itera per ogni categoria
    for class_id, label in categories.items():
        files = video_labels[video_labels == class_id].index.tolist()
        count = len(files)
        
        # Conteggio persone per questa specifica categoria
        mask = df['video_id'].isin(files)
        persons_in_cat = len(df[mask].groupby(['video_id', 'person_id']))
        
        print(f"\n>>> {label} | Video: {count} ({count/total_videos*100:.1f}%) | Persone: {persons_in_cat}")
        print("-" * 40)
        
        if count > 0:
            for f in sorted(files):
                # Conta persone nel singolo video
                pers_in_video = len(df[df['video_id'] == f].groupby('person_id'))
                print(f"  • {f:<40} ({pers_in_video} persone)")
        else:
            print("  (Nessun video trovato)")

    print("\n" + "="*60 + "\n")
'''
# Creazione cartelle se non esistono
for folder in [OUTPUT_DIR, OUTPUT_DIR + "/text_report", OUTPUT_DIR + "/confusion_matrix/by_video"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ----- FRAME CONSECUTIVI -----
def has_consecutive_violence(group, threshold=5):
    group = group.sort_values("frame_id")
    is_violent = group["pred_class"] > 0
    consecutive_groups = is_violent.groupby((is_violent != is_violent.shift()).cumsum())
    
    max_consecutive = 0
    for _, grp in consecutive_groups:
        if grp.iloc[0]: # Se è True (Violento)
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
    
    try:
        df = pd.read_csv(filename)
        print(f"Caricati {len(df)} frame totali da {filename}")
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return

    # Pulizia e casting
    df['pred_class'] = pd.to_numeric(df['pred_class'], errors='coerce').fillna(0).astype(int)
    df['true_class'] = pd.to_numeric(df['true_class'], errors='coerce').fillna(0).astype(int)
    
    # Score unificato (per eventuale AUC futura)
    df["unified_score"] = df.apply(lambda row: 1.0 if row.get("has_knife", 0) == 1 else row.get("violence_score", 0), axis=1)

    # Calcolo Flicker
    avg_flicker = calculate_flicker_rate(df)
    print(f"Flicker Rate medio del dataset: {avg_flicker:.4f}")

    # ----- PREPARAZIONE LISTE PER GRAFICO -----
    threshold_values = []
    precision_non_violent = [] # Classe 0
    precision_violent = []     # Classe 1
    precision_macro = []       # Media Unita (Macro Avg)

    # Raggruppamento base (ottimizzazione)
    grouped_base = list(df.groupby(['video_id'])) 

    # ----- LOOP SOGLIE (da 20 a 40) -----
    start_thresh = 20
    end_thresh = 40
    print(f"\nInizio analisi iterativa per soglie da {start_thresh} a {end_thresh}...")

    target_names = ["Non Violento", "Violento"]

    for thresh in range(start_thresh, end_thresh + 1):
        
        y_true_video = []
        y_pred_video = []
        y_score_video = [] 

        # Calcolo predizioni per questa soglia
        for video_id, video_group in grouped_base:
            
            # Ground Truth Video
            video_true_class = 1 if video_group['true_class'].max() > 0 else 0

            # Predizione Video
            video_is_predicted_violent = False
            people_in_video = video_group.groupby('person_id')
            
            for person_id, person_data in people_in_video:
                if has_consecutive_violence(person_data, threshold=thresh):
                    video_is_predicted_violent = True
                    break 
            
            video_pred_class = 1 if video_is_predicted_violent else 0
            
            y_true_video.append(video_true_class)
            y_pred_video.append(video_pred_class)
            y_score_video.append(video_group['unified_score'].max())

        y_true_video = np.array(y_true_video)
        y_pred_video = np.array(y_pred_video)

        # --- METRICHE & DATI GRAFICO ---
        report_dict = classification_report(y_true_video, y_pred_video, target_names=target_names, output_dict=True, zero_division=0)
        
        threshold_values.append(thresh)
        precision_non_violent.append(report_dict["Non Violento"]["precision"])
        precision_violent.append(report_dict["Violento"]["precision"])
        precision_macro.append(report_dict["macro avg"]["precision"]) # <--- LINEA "UNITI"

        # --- SALVATAGGIO REPORT TXT ---
        txt_filename = os.path.join(OUTPUT_DIR, OUTPUT_TXT_FILE+f"_{thresh}.txt")
        with open(txt_filename, "w") as f:
            f.write(f"=== REPORT SOGLIA: {thresh} ===\n")
            f.write(classification_report(y_true_video, y_pred_video, target_names=target_names, zero_division=0))
            if len(np.unique(y_true_video)) > 1:
                roc = roc_auc_score(y_true_video, y_score_video)
                f.write(f"\nROC-AUC Score: {roc:.4f}\n")
            f.write(f"Flicker Rate: {avg_flicker:.4f}\n")

        # --- SALVATAGGIO MATRICE CONFUSIONE (NO SHOW) ---
        cm = confusion_matrix(y_true_video, y_pred_video)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'CM - Threshold {thresh}')
        plt.ylabel('Reale')
        plt.xlabel('Predetto')
        plt.tight_layout()
        
        cm_filename = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE+f"_{thresh}.png")
        plt.savefig(cm_filename)
        plt.close() # <--- IMPORTANTE: Chiude il plot senza mostrarlo per non bloccare il loop
        
        print(f" > Soglia {thresh}: Precision V = {report_dict['Violento']['precision']:.2f}, Macro = {report_dict['macro avg']['precision']:.2f}")

    # ----- GRAFICO FINALE COMPARATIVO -----
    print("\nGenerazione grafico finale...")
    
    plt.figure(figsize=(12, 7))
    
    # Linea Non Violento (Blu tratteggiata)
    plt.plot(threshold_values, precision_non_violent, marker='o', label='Non Violento', color='blue', linestyle=':', alpha=0.6)
    
    # Linea Violento (Rossa solida - Focus principale)
    plt.plot(threshold_values, precision_violent, marker='o', label='Violento', color='red', linewidth=2)

    # Linea Uniti / Macro Average (Verde tratteggiata)
    plt.plot(threshold_values, precision_macro, marker='s', label='Combined (Macro Avg)', color='green', linestyle='--', linewidth=2)
    
    plt.title('Precision Analysis: Threshold Selection (20-40 frames)')
    plt.xlabel('Consecutive Frames Threshold')
    plt.ylabel('Precision Score')
    plt.xticks(threshold_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    final_curve_path = os.path.join(OUTPUT_DIR, "precision_curve_comparison.png")
    plt.savefig(final_curve_path, bbox_inches='tight')
    print(f"[INFO] Grafico completo salvato in: {final_curve_path}")
    print(f"[INFO] Matrici di confusione salvate")
    
    plt.show()

if __name__ == "__main__":
    main()