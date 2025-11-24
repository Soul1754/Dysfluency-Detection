import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from sklearn.metrics import classification_report
import gc

# --- CONFIGURATION ---
CONFIG = {
    'data_dir': 'dataset/processed',
    # CHANGED: Using the Cross-Lingual (XLSR) Large model
    'model_name': "facebook/wav2vec2-large-xlsr-53", 
    'num_classes': 5,
    'batch_size': 1,             # STRICTLY 1 for XLSR on 8GB RAM
    'grad_accum_steps': 16,      # Increased accum steps to stabilize the larger model
    'learning_rate': 5e-5,       # Lower LR is better for Large models
    'epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- DATASET (Unchanged) ---
class StutterDataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path)
        return data['waveform'], data['labels']

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return waveforms_padded, labels_padded

# --- UPDATED MODEL ---
class StutterDetector(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        print(f"Loading {model_name}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # 1. FREEZE Feature Extractor (CNN layers)
        # These layers process raw audio and are very memory intensive.
        # We freeze them because they are already excellent at edge detection.
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # 2. ENABLE GRADIENT CHECKPOINTING (Crucial for 8GB RAM + XLSR)
        # This saves memory by not storing all intermediate activations.
        if CONFIG['device'] == 'cuda':
            self.wav2vec2.gradient_checkpointing_enable()
        
        # Classification Head
        # XLSR Hidden size is 1024 (Base was 768)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)

    def forward(self, input_values):
        # Forward pass through Wav2Vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Project to classes
        logits = self.classifier(hidden_states)
        return logits

# --- TRAINING LOOP ---
def train():
    # Force garbage collection before starting
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Running on {CONFIG['device']}")
    
    # Prepare Data
    all_files = glob.glob(os.path.join(CONFIG['data_dir'], "sample_*.pt"))
    if not all_files:
        raise FileNotFoundError("No .pt files found. Run preprocessing.py first.")
        
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    train_ds = StutterDataset(CONFIG['data_dir'], train_files)
    val_ds = StutterDataset(CONFIG['data_dir'], val_files)
    
    # Num_workers=0 to avoid multi-process memory overhead on 8GB RAM
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialize Model
    model = StutterDetector(CONFIG['model_name'], CONFIG['num_classes']).to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Load Class Weights
    weights_path = os.path.join(CONFIG['data_dir'], "class_weights.pt")
    if os.path.exists(weights_path):
        class_weights = torch.load(weights_path).to(CONFIG['device'])
        print("Loaded computed class weights.")
    else:
        class_weights = None
        print("Warning: No class weights found.")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad() # Initialize gradient
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        for i, (audio, labels) in enumerate(train_loader):
            audio = audio.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            # Forward
            logits = model(audio)
            
            # Align shapes
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
            
            # Loss
            loss = criterion(logits.reshape(-1, CONFIG['num_classes']), labels.reshape(-1))
            loss = loss / CONFIG['grad_accum_steps']
            
            # Backward
            loss.backward()
            
            # Update Weights every X steps
            if (i + 1) % CONFIG['grad_accum_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * CONFIG['grad_accum_steps']
            
            # Print less frequently to reduce I/O
            if i % 500 == 0:
                print(f"  Step {i}, Loss: {loss.item() * CONFIG['grad_accum_steps']:.4f}")
            
            # Explicitly delete variables to free graph memory
            del audio, labels, logits, loss

        print(f"Avg Train Loss: {total_loss/len(train_loader):.4f}")
        
        # Validate
        validate(model, val_loader)
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"xlsr_stutter_epoch_{epoch}.pth")

def validate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Validating...")
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            logits = model(audio)
            min_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
            
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
            
            del audio, labels, logits
            
    print(classification_report(all_labels, all_preds, zero_division=0))

if __name__ == "__main__":
    train()