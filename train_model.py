import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
from glob import glob # Aggiungi questa libreria

# Ignora i warning per un output più pulito
warnings.filterwarnings("ignore")

# Definisci il dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Definizione del Dataset Personalizzato ---
class PoseDataset(Dataset):
    def __init__(self, data_list, max_seq_len=60, num_keypoints=25):
        self.data_list = data_list
        self.max_seq_len = max_seq_len
        self.num_keypoints = num_keypoints

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx] # Ora l'input è il percorso della cartella, non del file

        # Cerca tutti i file JSON all'interno della cartella
        try:
            json_files = sorted(glob(os.path.join(folder_path, '*.json')))
        except Exception:
            print(f"Warning: Could not find any json files in {folder_path}. Skipping.")
            return None, None

        if not json_files:
            print(f"Warning: No keypoint files found in {folder_path}. Skipping.")
            return None, None

        keypoints = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    frame_data = json.load(f)
                    if 'people' in frame_data and len(frame_data['people']) > 0:
                        body_parts = frame_data['people'][0]['pose_keypoints_2d']
                        # Estrai solo le coordinate (x, y) e ignorare la confidenza (score)
                        coords = [(body_parts[i], body_parts[i+1]) for i in range(0, len(body_parts), 3)]
                        keypoints.append(coords)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Could not load or parse file {file_path}. Skipping.")
                continue

        if not keypoints:
            print(f"Warning: No keypoints found in {folder_path} after processing. Skipping.")
            return None, None

        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Pad o Trunca la sequenza per avere una lunghezza fissa
        seq_len = keypoints.shape[0]
        if seq_len > self.max_seq_len:
            keypoints = keypoints[:self.max_seq_len]
        else:
            padding = np.zeros((self.max_seq_len - seq_len, self.num_keypoints, 2), dtype=np.float32)
            keypoints = np.vstack([keypoints, padding])
        
        # Appiattisci le coordinate (x, y) di ogni keypoint per ogni frame
        # La forma finale sarà (max_seq_len, num_keypoints * 2)
        keypoints = keypoints.reshape(self.max_seq_len, -1)
        keypoints = torch.from_numpy(keypoints).float()
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return keypoints, label_tensor

# --- 2. Definizione del Modello CNN+LSTM in PyTorch ---
class ActionRecognitionModel(nn.Module):
    # Il codice di questa classe rimane identico
    def __init__(self, num_classes, num_keypoints=25, hidden_size=128):
        super(ActionRecognitionModel, self).__init__()
        
        # CNN per l'estrazione delle features da ogni frame
        # Usiamo un layer lineare (FC) per elaborare i keypoint di un singolo frame,
        # che è una soluzione più adatta della Conv1d per questo tipo di dati.
        self.cnn = nn.Sequential(
            nn.Linear(num_keypoints * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # LSTM per l'analisi della sequenza temporale
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layer per la classificazione
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Riorganizza l'input per la CNN
        # x: (batch, seq_len, num_keypoints*2) -> (batch*seq_len, num_keypoints*2)
        x = x.view(batch_size * seq_len, -1)
        
        # Applica la CNN ad ogni frame
        cnn_out = self.cnn(x)
        
        # Riorganizza l'output per l'LSTM
        # (batch*seq_len, 128) -> (batch, seq_len, 128)
        lstm_input = cnn_out.view(batch_size, seq_len, -1)
        
        # Applica l'LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Prende l'output dell'ultimo step
        last_step_out = lstm_out[:, -1, :]
        
        # Applica il classificatore finale
        output = self.fc(last_step_out)
        
        return output

# --- 3. Caricamento dei Dati e Addestramento ---
if __name__ == "__main__":
    # La logica di caricamento dei dati è leggermente diversa
    violent_path = "json_output/violent"
    non_violent_path = "json_output/non-violent"
    
    if not os.path.exists(violent_path) or not os.path.exists(non_violent_path):
        print("Error: Dataset directories not found. Please check your path configuration.")
        exit()
        
    data_list = []
    
    # Processa il file delle azioni violente
    try:
        with open('violent.txt', 'r') as f:
            for line in f:
                file_name, actions = line.strip().split(';')
                video_name = file_name.replace('.mp4', '')
                if 'stab' in actions:
                    for cam in ['cam1', 'cam2']:
                        # Ora aggiungi il percorso della CARTELLA di output, non del file .json
                        data_list.append((os.path.join(violent_path, cam, video_name), 'stab'))
                else:
                    for cam in ['cam1', 'cam2']:
                        data_list.append((os.path.join(violent_path, cam, video_name), 'other_violent'))
    except FileNotFoundError:
        print("Error: violent.txt not found.")
        exit()

    # Processa il file delle azioni non violente
    try:
        with open('non-violent.txt', 'r') as f:
            for line in f:
                file_name, _ = line.strip().split(';')
                video_name = file_name.replace('.mp4', '')
                for cam in ['cam1', 'cam2']:
                    data_list.append((os.path.join(non_violent_path, cam, video_name), 'non-violent'))
    except FileNotFoundError:
        print("Error: non-violent.txt not found.")
        exit()
        
    labels = [item[1] for item in data_list]
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    final_data_list = [(data_list[i][0], labels_encoded[i]) for i in range(len(data_list))]

    # Suddivisione del dataset in training e validation
    train_data, val_data = train_test_split(final_data_list, test_size=0.2, random_state=42)

    # Crea le istanze del Dataset e del DataLoader
    train_dataset = PoseDataset(train_data)
    val_dataset = PoseDataset(val_data)
    
    # Funzione per filtrare i batch che contengono None
    def collate_fn(batch):
        batch = [item for item in batch if item[0] is not None]
        if not batch:
            return None, None
        sequences = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return sequences, labels

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

    # Inizializza il modello
    num_classes = len(le.classes_)
    model = ActionRecognitionModel(num_classes).to(device)

    # Definizione della loss function e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- Ciclo di Addestramento ---
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            if sequences is None:
                continue
            
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # --- Valutazione sul set di validazione ---
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sequences, labels in val_loader:
                if sequences is None:
                    continue
                
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if total > 0:
                print(f'Validation Accuracy: {100 * correct / total:.2f}%')
            else:
                print('No valid data in validation set.')
            
    # Salva il modello addestrato
    torch.save(model.state_dict(), 'action_recognition_model_pytorch.pth')
    print("Modello salvato come 'action_recognition_model_pytorch.pth'")