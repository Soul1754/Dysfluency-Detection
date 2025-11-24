import os
import glob
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from sklearn.metrics import classification_report
import numpy as np

# --- CONFIGURATION ---
CONFIG = {
    'data_dir': 'dataset/processed',
    'model_name': "facebook/wav2vec2-base", 
    'num_classes': 5,
    'batch_size': 4,
    'grad_accum_steps': 8,
    'learning_rate': 5e-5,       # Lowered slightly for stability
    'epochs': 10,
    'max_duration_sec': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class StutterDataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            data = torch.load(path, weights_only=True)
            waveform = data['waveform']
            labels = data['labels']

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            max_len = 16000 * CONFIG['max_duration_sec']
            if waveform.size(1) > max_len:
                waveform = waveform[:, :max_len]
                labels = labels[:max_len] # Note: This label slice is approximate, handled in collate
            
            return waveform, labels
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy data to prevent crash
            return torch.zeros(1, 16000), torch.zeros(50, dtype=torch.long)

def collate_fn(batch):
    # Filter out bad loads
    batch = [item for item in batch if item[0].sum() != 0]
    if not batch: return torch.tensor([]), torch.tensor([])

    waveforms = [item[0].squeeze(0) for item in batch]
    labels = [item[1] for item in batch]
    
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return waveforms_padded, labels_padded

# --- MODEL ---
class StutterDetector(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.wav2vec2.feature_extractor._freeze_parameters()
        if CONFIG['device'] == 'cuda':
            self.wav2vec2.gradient_checkpointing_enable()

        # Add a Dropout layer to prevent overfitting to Class 0
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

def train():
    gc.collect()
    torch.cuda.empty_cache()
    
    # 1. LOAD WEIGHTS (CRITICAL FIX)
    weights_path = os.path.join(CONFIG['data_dir'], "class_weights.pt")
    if os.path.exists(weights_path):
        class_weights = torch.load(weights_path).to(CONFIG['device'])
        print(f"Loaded Class Weights: {class_weights}")
    else:
        print("WARNING: No class weights found. Model will likely collapse to Class 0.")
        class_weights = None

    # 2. Prepare Data
    all_files = glob.glob(os.path.join(CONFIG['data_dir'], "sample_*.pt"))
    split_idx = int(len(all_files) * 0.8)
    train_loader = DataLoader(StutterDataset(CONFIG['data_dir'], all_files[:split_idx]), 
                              batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(StutterDataset(CONFIG['data_dir'], all_files[split_idx:]), 
                            batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 3. Setup
    model = StutterDetector(CONFIG['model_name'], CONFIG['num_classes']).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scaler = torch.amp.GradScaler('cuda')
    
    # Apply weights to loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    print("Starting training...")

    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (audio, labels) in enumerate(train_loader):
            if audio.numel() == 0: continue # Skip empty batches
            
            audio = audio.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            with torch.amp.autocast('cuda'):
                logits = model(audio)
                
                # Align lengths
                # Wav2Vec2 output is slightly shorter than Input/320 due to conv padding
                target_len = logits.shape[1]
                if labels.shape[1] > target_len:
                    labels = labels[:, :target_len]
                elif labels.shape[1] < target_len:
                    logits = logits[:, :labels.shape[1], :]

                loss = criterion(logits.reshape(-1, CONFIG['num_classes']), labels.reshape(-1))
                loss = loss / CONFIG['grad_accum_steps']

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG['grad_accum_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * CONFIG['grad_accum_steps']
            del audio, labels, logits, loss

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # Validate every epoch
        validate(model, val_loader)
        
        torch.save(model.state_dict(), f"./models/stutter_model_epoch_{epoch+1}.pth")

def validate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Validating...")
    with torch.no_grad():
        for audio, labels in loader:
            if audio.numel() == 0: continue
            
            audio = audio.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            # Forward pass
            logits = model(audio)
            
            # Align lengths (Crucial for Wav2Vec2)
            target_len = logits.shape[1]
            if labels.shape[1] > target_len:
                labels = labels[:, :target_len]
            elif labels.shape[1] < target_len:
                logits = logits[:, :labels.shape[1], :]
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Filter out padded tokens (-100)
            mask = labels != -100
            
            # Move to CPU and extend lists
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    # --- FIX IS HERE ---
    # We explicitly provide labels=[0, 1, 2, 3, 4]
    # This tells sklearn: "Even if you only see '0', look for all 5 classes."
    print("\nClassification Report:")
    try:
        print(classification_report(
            all_labels, 
            all_preds, 
            zero_division=0, 
            labels=[0, 1, 2, 3, 4],  # <--- THIS PREVENTS THE CRASH
            target_names=["Fluent", "Block", "WordRep", "SylRep", "Prolong"]
        ))
    except Exception as e:
        print(f"Report generation failed: {e}")
        print(f"Unique Labels found: {set(all_labels)}")
        print(f"Unique Preds found: {set(all_preds)}")
    print("-" * 30)

if __name__ == "__main__":
    train()