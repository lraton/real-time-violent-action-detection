import os
import subprocess
from tqdm import tqdm

# Impostazioni di configurazione
# Percorso completo dell'eseguibile di OpenPose Portable
OPENPOSE_EXECUTABLE = r'D:\Download\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\bin\OpenPoseDemo.exe'

# Cartella radice del tuo dataset di video
VIDEO_ROOT_FOLDER = 'video_dataset'

# Cartella dove verranno salvati i file JSON
OUTPUT_ROOT_FOLDER = 'json_output'

def run_openpose_on_video(video_path, output_json_path):
    """
    Esegue OpenPose su un singolo file video e salva i keypoint in formato JSON.
    """
    if not os.path.exists(os.path.dirname(output_json_path)):
        os.makedirs(os.path.dirname(output_json_path))

    # Estrae il percorso della cartella principale di OpenPose
    openpose_root_folder = os.path.dirname(os.path.dirname(OPENPOSE_EXECUTABLE))
    
    # Converte i percorsi relativi in percorsi assoluti per evitare errori
    absolute_video_path = os.path.abspath(video_path)
    absolute_output_path = os.path.abspath(output_json_path)
    
    # Comando per eseguire OpenPose
    command = [
        OPENPOSE_EXECUTABLE,
        '--video', absolute_video_path,
        '--write_json', absolute_output_path,
        '--display', '0',
        '--render_pose', '0'
    ]

    try:
        # Esegue il comando specificando la directory di lavoro di OpenPose
        subprocess.run(command, check=True, cwd=openpose_root_folder)
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'elaborazione di {video_path}: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Errore: L'eseguibile di OpenPose non è stato trovato al percorso specificato.")
        return False
    return True

def main():
    """
    Scansiona la struttura delle cartelle, trova tutti i video e li elabora con OpenPose.
    """
    print("Inizio la pre-elaborazione dei video con OpenPose...")
    
    video_files = []
    for root, _, files in os.walk(VIDEO_ROOT_FOLDER):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print("⚠️ Nessun file video trovato. Controlla il percorso 'VIDEO_ROOT_FOLDER'.")
        return

    for video_path in tqdm(video_files, desc="Elaborazione video"):
        relative_path = os.path.relpath(video_path, VIDEO_ROOT_FOLDER)
        output_folder = os.path.join(OUTPUT_ROOT_FOLDER, os.path.dirname(relative_path))
        
        output_file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_json_path = os.path.join(output_folder, output_file_name)
        
        # OpenPose crea una cartella per i JSON del video, non un singolo file
        if os.path.exists(output_json_path) and os.listdir(output_json_path):
            continue
        
        run_openpose_on_video(video_path, output_json_path)

    print("\n✅ Pre-elaborazione completata per tutti i video.")

if __name__ == '__main__':
    main()