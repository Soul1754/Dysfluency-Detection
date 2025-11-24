import torch
import torchaudio
import numpy as np
import librosa
from transformers import Wav2Vec2Model

# Import the model class structure from your train.py 
# (Assuming train.py is importable, otherwise paste the StutterDetector class here)
try:
    from train import StutterDetector
except ImportError:
    # Fallback if train.py isn't in path, paste the class definition:
    import torch.nn as nn
    class StutterDetector(nn.Module):
        def __init__(self, model_name, num_classes):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
            self.wav2vec2.feature_extractor._freeze_parameters()
            self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)
        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            logits = self.classifier(outputs.last_hidden_state)
            return logits

class StutterPredictor:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Model
        print(f"Loading model from {model_path} on {self.device}...")
        self.model = StutterDetector(config['model_name'], config['num_classes'])
        
        # Load Weights (Handle generic state dict vs full checkpoint)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define Class Mapping (Inverse of your config)
        self.id2label = {v: k for k, v in config['classes'].items()}

    def _preprocess_audio(self, audio_input, orig_sr):
        """
        Standardizes input: Resample to 16k, Mono, Normalize
        """
        # 1. Convert to Tensor
        if isinstance(audio_input, np.ndarray):
            waveform = torch.tensor(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input.float()
        else:
            raise ValueError("Input must be Numpy Array or PyTorch Tensor")

        # 2. Handle Channels (Stereo to Mono)
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # Add channel dim: [1, samples]

        # 3. Resample to 16kHz
        if orig_sr != self.config['target_sr']:
            resampler = torchaudio.transforms.Resample(orig_sr, self.config['target_sr'])
            waveform = resampler(waveform)

        # 4. Normalize (Standardize amplitude)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        return waveform.squeeze() # Return [samples]

    def predict_file(self, file_path):
        """
        Method 1: Input via File Path
        """
        waveform, sr = torchaudio.load(file_path)
        return self.predict_long_audio(waveform, sr)

    def predict_array(self, audio_array, sampling_rate):
        """
        Method 2: Input via Memory (Numpy/Tensor)
        """
        return self.predict_long_audio(audio_array, sampling_rate)

    def predict_long_audio(self, audio_input, orig_sr):
        """
        Method 3: Sliding Window Strategy for Long Audio
        """
        # Preprocess to 16k 1D tensor
        waveform = self._preprocess_audio(audio_input, orig_sr)
        
        # Window settings
        window_size = 16000 * 5  # 5 seconds
        stride = 16000 * 5       # Non-overlapping for simplicity (can add overlap logic)
        
        all_events = []
        total_samples = len(waveform)
        
        print(f"Processing {total_samples/16000:.2f} seconds of audio...")

        with torch.no_grad():
            for start_idx in range(0, total_samples, stride):
                end_idx = min(start_idx + window_size, total_samples)
                
                # Extract chunk
                chunk = waveform[start_idx:end_idx]
                
                # Pad if chunk is too short (Wav2Vec2 needs min length)
                if len(chunk) < 1600: # Minimum 0.1s
                    continue
                    
                # Add batch dimension [1, samples]
                input_values = chunk.unsqueeze(0).to(self.device)

                # Inference
                logits = self.model(input_values)
                preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                
                # Convert frame indices to time
                # Wav2Vec2 output stride is approx 320 samples (20ms)
                model_stride_samples = 320 
                
                current_event = None
                
                for i, label_id in enumerate(preds):
                    label_name = self.id2label[label_id]
                    
                    # Skip 'fluent' (id 0)
                    if label_id == 0:
                        if current_event:
                            # Close previous event
                            current_event['end'] = (start_idx + (i * model_stride_samples)) / 16000.0
                            all_events.append(current_event)
                            current_event = None
                        continue
                    
                    # Start new event
                    if current_event is None:
                        timestamp = (start_idx + (i * model_stride_samples)) / 16000.0
                        current_event = {
                            'type': label_name,
                            'start': timestamp,
                            'end': timestamp + 0.02 # Min duration
                        }
                    # If label changed (e.g. block -> repetition)
                    elif current_event['type'] != label_name:
                        # Close old
                        current_event['end'] = (start_idx + (i * model_stride_samples)) / 16000.0
                        all_events.append(current_event)
                        # Start new
                        timestamp = (start_idx + (i * model_stride_samples)) / 16000.0
                        current_event = {
                            'type': label_name,
                            'start': timestamp,
                            'end': timestamp + 0.02
                        }
                    # If same label, continue (do nothing, just extend implicitly)

                # Close any event lingering at end of chunk
                if current_event:
                     current_event['end'] = (start_idx + (len(preds) * model_stride_samples)) / 16000.0
                     all_events.append(current_event)

        return self._merge_close_events(all_events)

    def _merge_close_events(self, events, tolerance=0.1):
        """
        Clean up: Merges stutter events that are very close together
        (e.g. two 'blocks' separated by 0.05s likely belong to same event)
        """
        if not events: return []
        
        merged = [events[0]]
        for current in events[1:]:
            prev = merged[-1]
            
            # If same type and close enough
            if current['type'] == prev['type'] and (current['start'] - prev['end'] < tolerance):
                prev['end'] = current['end'] # Extend
            else:
                merged.append(current)
        return merged

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Configuration matching your train.py
    CONFIG = {
        'model_name': "facebook/wav2vec2-base",
        'target_sr': 16000,
        'num_classes': 5,
        'classes': {
            'fluent': 0, 'block': 1, 'word_rep': 2, 'syllab_rep': 3, 'prolongation': 4
        }
    }

    # Initialize (Point to your saved .pth file)
    # Ensure you have a file named 'stutter_base_epoch_1.pth' or similar
    try:
        predictor = StutterPredictor("stutter_base_epoch_1.pth", CONFIG)
        
        # Method 1 Example: File
        # events = predictor.predict_file("test_audio.wav")
        # print("Events found:", events)
        
        # Method 2 Example: Dummy Array
        dummy_audio = np.random.uniform(-0.5, 0.5, 48000) # 1 sec of noise
        events = predictor.predict_array(dummy_audio, 48000)
        print("Dummy Events:", events)
        
    except FileNotFoundError:
        print("Please train the model first to generate the .pth file!")