import torch
import torchaudio
from transformers import Wav2Vec2Model
import torch.nn as nn
import os
import soundfile as sf  # <--- NEW IMPORT
import numpy as np

# --- 1. DEFINE MODEL ---
class StutterDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Ensure we load the correct config
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        return self.classifier(outputs.last_hidden_state)

# --- 2. LOAD MODEL ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

model = StutterDetector().to(device)
model_path = "./models/stutter_model_epoch_6.pth"  # Ensure this path is correct

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit()

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# --- 3. PREPARE AUDIO (THE FIX) ---
audio_path = r"E:\College\Final Y\Sem I\EDAI\stuttering\dataset\wav_files_16k\audio_0009_prolongation_tonic.wav"

if not os.path.exists(audio_path):
    print(f"Error: Audio file not found at {audio_path}")
    exit()

print(f"Loading {os.path.basename(audio_path)}...")

# --- REPLACEMENT FOR TORCHAUDIO.LOAD ---
try:
    # 1. Read file using SoundFile (Works reliably on Windows)
    audio_signal, sr = sf.read(audio_path)
    
    # 2. Convert to Tensor
    waveform = torch.from_numpy(audio_signal).float()
    
    # 3. Handle Shape: SoundFile is [Time, Channels], PyTorch needs [Channels, Time]
    if waveform.dim() == 1:
        # Mono: [Time] -> [1, Time]
        waveform = waveform.unsqueeze(0)
    else:
        # Stereo: [Time, Channels] -> [Channels, Time]
        waveform = waveform.t()

except Exception as e:
    print(f"Failed to load audio: {e}")
    exit()

# --- RESAMPLING ---
target_sr = 16000
if sr != target_sr:
    print(f"Resampling from {sr} to {target_sr}Hz...")
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
    # Move to GPU for resampling if available
    waveform = waveform.to(device)
    waveform = resampler(waveform)
else:
    waveform = waveform.to(device)

# --- 4. INFERENCE ---
classes = ['Fluent', 'Block', 'WordRep', 'SylRep', 'Prolongation']

with torch.no_grad():
    # Model expects [Batch, Time]. We have [Channels, Time].
    # If mono, we are good (it acts as batch 1). 
    # If stereo, pick first channel.
    if waveform.size(0) > 1:
        waveform = waveform[0].unsqueeze(0)
        
    logits = model(waveform)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(logits, dim=-1)

# Move to CPU for printing
preds = preds[0].cpu().numpy()
probs = probs[0].cpu().numpy()

# --- 5. PRINT RESULTS ---
print(f"\nAnalysis for: {os.path.basename(audio_path)}")
print(f"Total Frames: {len(preds)}")
print("-" * 40)

found_stutter = False
for i, p in enumerate(preds):
    if p != 0: # If NOT fluent
        found_stutter = True
        confidence = probs[i][p]
        time_sec = (i * 320) / 16000 
        print(f"Time: {time_sec:.2f}s | Detected: {classes[p]:<12} | Conf: {confidence:.2f}")

if not found_stutter:
    print("Result: Fluent speech detected (No stuttering found).")