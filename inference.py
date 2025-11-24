import torch
import torchaudio
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2Model
import os
import soundfile


# --- SHARED CLASS DEFINITION ---
# This must match the class definition used in train.py exactly.
class StutterDetector(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        # Freeze feature extractor to match training configuration
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        # We use the last hidden state for frame-level classification
        logits = self.classifier(outputs.last_hidden_state)
        return logits

class StutterPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # torchaudio.set_audio_backend("soundfile")
        print(f"Loading checkpoint from {model_path} on {self.device}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}. Make sure you have trained the model first.")
        
        # 1. Load Configuration from the file
        # If 'config' exists in the checkpoint, we use it. Otherwise, we fall back to defaults.
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            print("Warning: Config not found in checkpoint. Using default configuration.")
            self.config = {
                'model_name': "facebook/wav2vec2-base",
                'num_classes': 5,
                'target_sr': 16000
            }

        # Ensure target_sr is set in config if it wasn't saved
        if 'target_sr' not in self.config:
            self.config['target_sr'] = 16000

        # 2. Initialize Model Architecture
        self.model = StutterDetector(
            self.config['model_name'], 
            self.config['num_classes']
        )
        
        # 3. Load Weights
        # Handle case where the file is a full checkpoint dict vs just the state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assumes the file is just the state_dict
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Map IDs to Labels
        # Ideally, this should be saved in the config during training. 
        # Here we default to the standard mapping.
        self.id2label = {
            0: 'fluent', 
            1: 'block', 
            2: 'word_rep', 
            3: 'syllab_rep', 
            4: 'prolongation'
        }

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
        target_sr = self.config['target_sr']
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)

        # 4. Normalize (Standardize amplitude)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        return waveform.squeeze() # Return [samples]

    def predict_file(self, file_path):
        """
        Method 1: Input via File Path (Robust Version)
        """
        try:
            # Attempt 1: Standard load
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Attempt 2: Direct load via soundfile (Bypasses backend errors)
            print(f"Torchaudio load failed ({e}), falling back to soundfile direct load...")
            import soundfile as sf
            
            # soundfile reads as [samples, channels]
            audio_np, sr = sf.read(file_path) 
            waveform = torch.tensor(audio_np).float()
            
            # Convert to [channels, samples] for consistency
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0) # Add channel dim
            else:
                waveform = waveform.t() # Transpose to [channels, samples]
                
        return self.predict_long_audio(waveform, sr)

    def predict_array(self, audio_array, sampling_rate):
        """
        Method 2: Input via Memory (Numpy/Tensor)
        """
        return self.predict_long_audio(audio_array, sampling_rate)

    # def predict_long_audio(self, audio_input, orig_sr):
    #     """
    #     Method 3: Sliding Window Strategy for Long Audio
    #     """
    #     # Preprocess to 16k 1D tensor
    #     waveform = self._preprocess_audio(audio_input, orig_sr)
        
    #     # Window settings
    #     target_sr = self.config['target_sr']
    #     window_size = target_sr * 5  # 5 seconds
    #     stride = target_sr * 5       # Non-overlapping for simplicity
        
    #     all_events = []
    #     total_samples = len(waveform)
        
    #     # print(f"Processing {total_samples/target_sr:.2f} seconds of audio...")

    #     with torch.no_grad():
    #         for start_idx in range(0, total_samples, stride):
    #             end_idx = min(start_idx + window_size, total_samples)
                
    #             # Extract chunk
    #             chunk = waveform[start_idx:end_idx]
                
    #             # Pad if chunk is too short (Wav2Vec2 needs min length)
    #             if len(chunk) < 1600: # Minimum 0.1s
    #                 continue
                    
    #             # Add batch dimension [1, samples]
    #             input_values = chunk.unsqueeze(0).to(self.device)

    #             # Inference
    #             logits = self.model(input_values)
    #             preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                
    #             # Convert frame indices to time
    #             # Wav2Vec2 output stride is approx 320 samples (20ms at 16k)
    #             model_stride_samples = 320 
                
    #             current_event = None
                
    #             for i, label_id in enumerate(preds):
    #                 label_name = self.id2label[label_id]
                    
    #                 # Skip 'fluent' (id 0)
    #                 if label_id == 0:
    #                     if current_event:
    #                         # Close previous event
    #                         current_event['end'] = (start_idx + (i * model_stride_samples)) / float(target_sr)
    #                         all_events.append(current_event)
    #                         current_event = None
    #                     continue
                    
    #                 # Start new event
    #                 if current_event is None:
    #                     timestamp = (start_idx + (i * model_stride_samples)) / float(target_sr)
    #                     current_event = {
    #                         'type': label_name,
    #                         'start': timestamp,
    #                         'end': timestamp + 0.02 # Min duration placeholder
    #                     }
    #                 # If label changed (e.g. block -> repetition)
    #                 elif current_event['type'] != label_name:
    #                     # Close old
    #                     current_event['end'] = (start_idx + (i * model_stride_samples)) / float(target_sr)
    #                     all_events.append(current_event)
    #                     # Start new
    #                     timestamp = (start_idx + (i * model_stride_samples)) / float(target_sr)
    #                     current_event = {
    #                         'type': label_name,
    #                         'start': timestamp,
    #                         'end': timestamp + 0.02
    #                     }
    #                 # If same label, continue (do nothing, just extend implicitly)

    #             # Close any event lingering at end of chunk
    #             if current_event:
    #                   current_event['end'] = (start_idx + (len(preds) * model_stride_samples)) / float(target_sr)
    #                   all_events.append(current_event)

    #     return self._merge_close_events(all_events)

    def predict_long_audio(self, audio_input, orig_sr):
        from collections import Counter # Add this import for debugging

        # Preprocess to 16k 1D tensor
        waveform = self._preprocess_audio(audio_input, orig_sr)
        
        target_sr = self.config['target_sr']
        window_size = target_sr * 5  
        stride = target_sr * 5       
        
        all_events = []
        total_samples = len(waveform)
        
        print(f"\n--- DEBUG: Analyzing {total_samples/target_sr:.2f} seconds ---")

        with torch.no_grad():
            for start_idx in range(0, total_samples, stride):
                end_idx = min(start_idx + window_size, total_samples)
                chunk = waveform[start_idx:end_idx]
                
                if len(chunk) < 1600: 
                    continue
                    
                input_values = chunk.unsqueeze(0).to(self.device)

                # Inference
                logits = self.model(input_values)
                preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                
                # --- DEBUG PRINT START ---
                # This tells us exactly what the model "sees"
                counts = Counter(preds)
                print(f"Time {start_idx/target_sr:.1f}s - {end_idx/target_sr:.1f}s | Raw Predictions: {dict(counts)}")
                # --- DEBUG PRINT END ---

                model_stride_samples = 320 
                current_event = None
                
                for i, label_id in enumerate(preds):
                    label_name = self.id2label.get(label_id, f"Class_{label_id}")
                    
                    # IF THE MODEL PREDICTS 0 (FLUENT), WE SKIP
                    if label_id == 0:
                        if current_event:
                            current_event['end'] = (start_idx + (i * model_stride_samples)) / float(target_sr)
                            all_events.append(current_event)
                            current_event = None
                        continue
                    
                    if current_event is None:
                        timestamp = (start_idx + (i * model_stride_samples)) / float(target_sr)
                        current_event = {
                            'type': label_name,
                            'start': timestamp,
                            'end': timestamp + 0.02 
                        }
                    elif current_event['type'] != label_name:
                        current_event['end'] = (start_idx + (i * model_stride_samples)) / float(target_sr)
                        all_events.append(current_event)
                        timestamp = (start_idx + (i * model_stride_samples)) / float(target_sr)
                        current_event = {
                            'type': label_name,
                            'start': timestamp,
                            'end': timestamp + 0.02
                        }

                if current_event:
                      current_event['end'] = (start_idx + (len(preds) * model_stride_samples)) / float(target_sr)
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
    # Replace this with the path to your actual trained model file
    # The new training script outputs files like 'stutter_model_epoch_10.pth'
    model_path = "./models/stutter_model_epoch_10.pth"
    
    try:
        predictor = StutterPredictor(model_path)
        print("Model loaded successfully!")
        
        # Method 1 Example: File
        test_file = r"E:\College\Final Y\Sem I\EDAI\stuttering\dataset\wav_files_16k\audio_0009_prolongation_tonic.wav"
        if os.path.exists(test_file):
            events = predictor.predict_file(test_file)
            print("Events found:", events)
        
        # Method 2 Example: Dummy Array (for testing without a file)
        # print("\nRunning dummy inference test...")
        # dummy_audio = np.random.uniform(-0.5, 0.5, 48000) # 3 sec of noise
        # events = predictor.predict_array(dummy_audio, 48000)
        # print("Dummy Events (expect mostly fluent/empty for noise):", events)
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")