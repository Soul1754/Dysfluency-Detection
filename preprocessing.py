import os
import glob
import torch
import torchaudio
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import Counter
import gc

# --- CONFIGURATION ---
CONFIG = {
    'raw_data_dir': 'dataset/raw',      
    'output_dir': 'dataset/processed',      
    'target_sr': 16000,                  
    'model_stride': 320,                 
    'min_duration_sec': 0.1,
    'classes': {
        'fluent': 0,
        'block': 1,
        'word_rep': 2,
        'syllab_rep': 3,
        'prolongation': 4
    },
    'batch_size': 200 
}

# --- SETUP GPU ---
# We check if CUDA is available, otherwise fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Processing on: {DEVICE} " + (f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

def safe_load_audio(audio_data):
    """
    Safely extracts and flattens audio.
    """
    if audio_data is None:
        return None
    
    try:
        if isinstance(audio_data, dict) and 'array' in audio_data:
            audio_data = audio_data['array']
        
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)
        
        if isinstance(audio_data, np.ndarray) and audio_data.dtype == 'object':
            audio_data = np.array(audio_data.tolist())

        if isinstance(audio_data, np.ndarray):
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            return audio_data.astype(np.float32)
            
        return None
    except Exception:
        return None

def process_dataset():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    global_label_counts = Counter()
    file_paths = glob.glob(os.path.join(CONFIG['raw_data_dir'], "*.parquet"))
    file_counter = 0
    skip_stats = Counter()
    skipped_log_path = os.path.join(CONFIG['output_dir'], "skipped_log.txt")
    
    with open(skipped_log_path, "w") as f:
        f.write("--- SKIPPED FILES LOG ---\n")

    # --- GPU RESAMPLER CACHE ---
    current_resampler = None
    current_resampler_sr = -1

    for pq_file in file_paths:
        print(f"Processing {pq_file}...")
        
        try:
            parquet_file = pq.ParquetFile(pq_file)
            batch_iter = parquet_file.iter_batches(batch_size=CONFIG['batch_size'])
            
            for batch in tqdm(batch_iter, desc="Batches", total=parquet_file.num_row_groups):
                df_batch = batch.to_pandas()

                for idx, row in df_batch.iterrows():
                    
                    # 1. Load (CPU)
                    raw_audio = safe_load_audio(row.get('audio'))
                    if raw_audio is None or len(raw_audio) == 0:
                        skip_stats['missing/empty'] += 1
                        continue

                    if np.isnan(raw_audio).any() or np.isinf(raw_audio).any():
                        skip_stats['corrupt_values'] += 1
                        continue

                    # 2. Move to GPU immediately
                    # torch.from_numpy creates a tensor sharing memory, then .to(DEVICE) moves it to VRAM
                    waveform = torch.from_numpy(raw_audio).to(DEVICE)
                    
                    orig_sr = 48000
                    if 'audio' in row and isinstance(row['audio'], dict):
                        orig_sr = row['audio'].get('sampling_rate', 48000)
                    
                    if orig_sr <= 0:
                        skip_stats['bad_sr'] += 1
                        continue

                    # 3. Resample (ON GPU)
                    if orig_sr != CONFIG['target_sr']:
                        if current_resampler is None or current_resampler_sr != orig_sr:
                            # Create resampler and move it to GPU
                            current_resampler = torchaudio.transforms.Resample(
                                orig_freq=orig_sr, 
                                new_freq=CONFIG['target_sr']
                            ).to(DEVICE)
                            current_resampler_sr = orig_sr
                        
                        try:
                            waveform = current_resampler(waveform)
                        except Exception:
                            skip_stats['resample_error'] += 1
                            continue

                    if waveform.size(0) < CONFIG['model_stride']:
                        skip_stats['too_short'] += 1
                        continue
                    
                    # 4. Normalize (ON GPU)
                    max_val = waveform.abs().max()
                    if max_val > 0:
                        waveform = waveform / max_val
                    else:
                        skip_stats['pure_silence'] += 1
                        continue

                    # 5. Labels (CPU Logic is fine here)
                    num_frames = int(waveform.size(0) / CONFIG['model_stride'])
                    labels = torch.zeros(num_frames, dtype=torch.long)
                    
                    # --- FIX START ---
                    # 1. Extract stutter type from the 'class' column
                    stutter_type = row.get('class') 
                    
                    # 2. Extract timestamps from the 'metadata' column (which is a dictionary)
                    meta = row.get('metadata')
                    start_t = None
                    end_t = None
                    
                    if isinstance(meta, dict):
                        start_t = meta.get('start_time')
                        end_t = meta.get('end_time')
                    # --- FIX END ---

                    # Check if extraction worked and matches our config
                    if stutter_type in CONFIG['classes'] and start_t is not None and end_t is not None:
                        # Validity Check
                        if start_t <= end_t and start_t >= 0:
                            class_id = CONFIG['classes'][stutter_type]
                            start_frame = int((start_t * CONFIG['target_sr']) / CONFIG['model_stride'])
                            end_frame = int((end_t * CONFIG['target_sr']) / CONFIG['model_stride'])
                            
                            start_frame = max(0, start_frame)
                            end_frame = min(num_frames, end_frame)
                            
                            if end_frame > start_frame:
                                labels[start_frame:end_frame] = class_id

                    # Stats
                    unique, counts = torch.unique(labels, return_counts=True)
                    for u, c in zip(unique, counts):
                        global_label_counts[int(u)] += int(c)

                    # 6. Save (Move back to CPU)
                    # Important: .cpu() moves it back to RAM so we can save it safely
                    output_path = os.path.join(CONFIG['output_dir'], f"sample_{file_counter}.pt")
                    torch.save({
                        'waveform': waveform.cpu(), 
                        'labels': labels      
                    }, output_path)
                    
                    file_counter += 1
                
                del df_batch
                gc.collect()

        except Exception as e:
            print(f"CRITICAL: Error reading {pq_file}: {e}")
            continue

    # --- SUMMARY ---
    print("\n" + "="*30)
    print("PROCESSING SUMMARY")
    print(f"Successfully Processed: {file_counter} samples")
    for reason, count in skip_stats.items():
        print(f" - {reason}: {count}")
    
    total_frames = sum(global_label_counts.values())
    weights = torch.ones(len(CONFIG['classes']))
    if total_frames > 0:
        for cls_name, cls_id in CONFIG['classes'].items():
            count = global_label_counts[cls_id]
            if count > 0:
                weights[cls_id] = total_frames / (len(CONFIG['classes']) * count)
    
    torch.save(weights, os.path.join(CONFIG['output_dir'], "class_weights.pt"))
    print("Done.")

if __name__ == "__main__":
    # This line is important for Windows GPU multiprocessing (though we aren't using MP here, it's good practice)
    torch.multiprocessing.set_start_method('spawn', force=True)
    process_dataset()