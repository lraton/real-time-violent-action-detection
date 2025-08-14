import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Definisci il dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Definizione del Dataset Personalizzato ---
class PoseDataset(Dataset):
    def __init__(self, data_list, max_seq_len=60, num_keypoints=25, transform=None):
        self.data_list = data_list
        self.max_seq_len = max_seq_len
        self.num_keypoints = num_keypoints
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        
        # Carica i keypoint dal file JSON
        try:
            with open(file_path, 'r') as f:
                keypoint_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load or parse file {file_path}. Skipping.")
            return None

        keypoints = []
        for frame in keypoint_data:
            # Assumiamo che ci sia una sola persona per frame
            if 'people' in frame and len(frame['people']) > 0:
                # Estrae le coordinate (x, y) dei keypoint
                body_parts = frame['people'][0]['pose_keypoints_2d']
                # Riorganizza da [x1, y1, c1, x2, y2, c2, ...] a [(x1, y1), (x2, y2), ...]
                coords = [(body_parts[i], body_parts[i+1]) for i in range(0, len(body_parts), 3)]
                keypoints.append(coords)
        
        if not keypoints:
            print(f"Warning: No keypoints found in {file_path}. Skipping.")
            return None

        # Converti in numpy array
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # Pad o Trunca la sequenza per avere una lunghezza fissa
        seq_len = keypoints.shape[0]
        if seq_len > self.max_seq_len:
            keypoints = keypoints[:self.max_seq_len]
        else:
            padding = np.zeros((self.max_seq_len - seq_len, self.num_keypoints, 2), dtype=np.float32)
            keypoints = np.vstack([keypoints, padding])
        
        # Appiattisci le dimensioni per l'input della CNN
        keypoints = keypoints.reshape(self.max_seq_len, -1, 1) # [T, H, W] per TimeDistributed CNN
        keypoints = torch.from_numpy(keypoints).float()
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return keypoints, label_tensor

# --- 2. Definizione del Modello CNN+LSTM in PyTorch ---
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes, num_keypoints=25, input_channels=1, hidden_size=128):
        super(ActionRecognitionModel, self).__init__()
        
        self.input_channels = input_channels
        self.num_keypoints = num_keypoints
        
        # CNN per l'estrazione delle features da ogni frame
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.num_keypoints * 2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # LSTM per l'analisi della sequenza temporale
        self.lstm = nn.LSTM(input_size=128 * 2, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layer per la classificazione
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _, _ = x.size()
        
        # Riorganizza l'input per la CNN
        # x: (batch, seq_len, num_keypoints, 2)
        x = x.view(batch_size * seq_len, -1) # -> (batch*seq_len, num_keypoints*2)
        
        # Applica la CNN ad ogni frame
        cnn_out = self.cnn(x.unsqueeze(2))
        
        # Riorganizza l'output per l'LSTM
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
    # Carica le liste dei file e le etichette
    violent_path = "dataset/violent"
    non_violent_path = "dataset/non-violent"
    
    data_list = []
    
    # Processa il file delle azioni violente
    with open('violent.txt', 'r') as f:
        for line in f:
            file_name, actions = line.strip().split(';')
            if 'stab' in actions:
                # 'stab' è la classe di interesse
                for cam in ['cam1', 'cam2']:
                    data_list.append((os.path.join(violent_path, cam, file_name.replace('.mp4', '.json')), 'stab'))
            else:
                # Tutte le altre azioni violente
                for cam in ['cam1', 'cam2']:
                    data_list.append((os.path.join(violent_path, cam, file_name.replace('.mp4', '.json')), 'other_violent'))

    # Processa il file delle azioni non violente
    with open('non-violent.txt', 'r') as f:
        for line in f:
            file_name, _ = line.strip().split(';')
            for cam in ['cam1', 'cam2']:
                data_list.append((os.path.join(non_violent_path, cam, file_name.replace('.mp4', '.json')), 'non-violent'))

    # Codifica delle etichette da stringa a numero
    labels = [item[1] for item in data_list]
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Aggiorna la lista dati con le etichette codificate
    final_data_list = [(data_list[i][0], labels_encoded[i]) for i in range(len(data_list))]

    # Filtra i dati non validi (files not found, etc.)
    final_data_list = [item for item in final_data_list if item is not None]

    # Suddivisione del dataset in training e validation
    train_data, val_data = train_test_split(final_data_list, test_size=0.2, random_state=42)

    # Crea le istanze del Dataset e del DataLoader
    train_dataset = PoseDataset(train_data)
    val_dataset = PoseDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

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
        for i, (sequences, labels) in enumerate(train_loader):
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
            
            print(f'Validation Accuracy: {100 * correct / total:.2f}%')
            
    # Salva il modello addestrato
    torch.save(model.state_dict(), 'action_recognition_model_pytorch.pth')
    print("Modello salvato come 'action_recognition_model_pytorch.pth'")