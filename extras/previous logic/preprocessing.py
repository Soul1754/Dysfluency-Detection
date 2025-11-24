# %%
# !pip install numpy pandas librosa tqdm pyarrow datasets

# %%
import os
import numpy as np
import pandas as pd
import librosa
import pyarrow.parquet as pq
from datasets import Dataset, Audio
from tqdm.auto import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')    

# %%
# Configuration
CONFIG = {
    'input_dir': './huggingface_models',  # Directory containing chunk-*.parquet files
    'output_dir': './processed_data',  # Output directory for processed chunks
    'target_sampling_rate': 16000,  # Wav2Vec2 requires 16kHz
    'frame_duration': 0.02,  # 20ms frames (50 fps)
    # Streaming/batching settings to limit RAM usage
    'stream_batch_size': 16,   # rows read per pyarrow batch (reduced for low-RAM machines)
    'save_batch_size': 32,     # processed samples buffered before saving to disk
    'stutter_classes': {
        'fluent': 0,
        'block': 1,
        'word_rep': 2,
        'syllab_rep': 3,
        'prolongation': 4
    }
}


# %%
def flatten_nested_audio(nested_array):
    """
    Flatten nested numpy arrays from parquet format
    Input: array([array([val1]), array([val2]), ...])
    Output: [val1, val2, ...]
    """
    try:
        if nested_array is None:
            return None
        
        # Handle different nesting structures
        if isinstance(nested_array, np.ndarray):
            if nested_array.dtype == object:
                # Nested array of arrays - flatten
                flattened = np.concatenate([arr.flatten() for arr in nested_array])
            else:
                # Already flat
                flattened = nested_array.flatten()
            
            return flattened.astype(np.float32)
        else:
            return None
    except Exception as e:
        print(f"Error flattening audio: {e}")
        return None

# %%
def resample_audio(audio_array, orig_sr, target_sr):
    """
    Resample audio from original sampling rate to target rate.
    Tries to use `resampy` (librosa's preferred backend). If `resampy` is not
    available it falls back to `scipy.signal.resample_poly` which is memory-
    and CPU-efficient for large arrays.
    """
    if orig_sr == target_sr:
        return audio_array
    
    # Prefer librosa/resampy when available for best quality
    try:
        import resampy  # optional dependency; librosa prefers this
        # If resampy is present, use librosa.resample (it will use resampy behind the scenes)
        resampled = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_best')
        return resampled
    except Exception:
        # Fallback: use scipy's resample_poly if available
        try:
            from scipy.signal import resample_poly
            # compute integer up/down factors using fractions for best ratio
            from fractions import Fraction
            frac = Fraction(int(target_sr), int(orig_sr)).limit_denominator()
            up, down = frac.numerator, frac.denominator
            resampled = resample_poly(audio_array, up, down)
            return resampled
        except Exception as e:
            print(f"Error resampling audio: {e}")
            print("To fix this, either install 'resampy' (`pip install resampy`) or 'scipy' (`pip install scipy`).")
            return None

# %%
def normalize_audio(audio_array):
    """
    Normalize audio to [-1, 1] range
    """
    if audio_array is None or len(audio_array) == 0:
        return None
    
    max_val = np.abs(audio_array).max()
    if max_val > 0:
        return audio_array / max_val
    return audio_array

# %%
def generate_frame_labels(audio_length, sampling_rate, start_time, end_time, stutter_class):
    """
    Generate frame-level labels for audio
    
    Args:
        audio_length: Number of audio samples
        sampling_rate: Audio sampling rate (Hz)
        start_time: Stutter start time (seconds)
        end_time: Stutter end time (seconds)
        stutter_class: Integer class ID
    
    Returns:
        Array of frame-level labels
    """
    # Calculate total duration and number of frames
    duration = audio_length / sampling_rate
    num_frames = int(duration / CONFIG['frame_duration'])
    
    # Initialize all frames as fluent (class 0)
    labels = np.zeros(num_frames, dtype=np.int64)
    
    # Mark stutter frames
    if start_time is not None and end_time is not None:
        start_frame = int(start_time / CONFIG['frame_duration'])
        end_frame = int(end_time / CONFIG['frame_duration'])
        
        # Ensure frames are within bounds
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        
        if start_frame < end_frame:
            labels[start_frame:end_frame] = stutter_class
    
    return labels

# %%
def process_single_sample(row):
    """
    Process a single data sample
    """
    try:
        # Extract audio data
        audio_dict = row['audio']
        if audio_dict is None or audio_dict.get('array') is None:
            return None
        
        # Flatten nested audio array
        audio_array = flatten_nested_audio(audio_dict['array'])
        if audio_array is None or len(audio_array) == 0:
            return None
        
        orig_sr = float(audio_dict.get('sampling_rate', 48000))
        
        # Resample to 16kHz
        audio_16khz = resample_audio(audio_array, orig_sr, CONFIG['target_sampling_rate'])
        if audio_16khz is None:
            return None
        
        # Normalize audio
        audio_normalized = normalize_audio(audio_16khz)
        if audio_normalized is None:
            return None
        
        # Extract metadata
        metadata = row.get('metadata', {})
        start_time = metadata.get('start_time')
        end_time = metadata.get('end_time')
        stutter_type = row.get('class', 'fluent')
        
        # Map stutter type to class ID
        stutter_class = CONFIG['stutter_classes'].get(stutter_type, 0)
        
        # Generate frame-level labels
        labels = generate_frame_labels(
            len(audio_normalized),
            CONFIG['target_sampling_rate'],
            start_time,
            end_time,
            stutter_class
        )
        
        # Calculate duration
        duration = len(audio_normalized) / CONFIG['target_sampling_rate']
        
        return {
            'audio': {
                'array': audio_normalized,
                'sampling_rate': CONFIG['target_sampling_rate']
            },
            'labels': labels.tolist(),
            'stutter_type': stutter_type,
            'duration': duration,
            'start_time': start_time if start_time is not None else 0.0,
            'end_time': end_time if end_time is not None else 0.0,
            'transcription': row.get('transcription', ''),
            'sentence': row.get('sentence', '')
        }
    
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

# %%
def process_parquet_chunk(parquet_path, chunk_name):
    """
    Stream-process a parquet file to limit memory usage.
    Reads the parquet in pyarrow batches, processes rows one-by-one,
    and saves intermediate HuggingFace dataset parts to disk so RAM is freed.
    """
    print(f"\n{'='*60}")
    print(f"Processing (stream): {chunk_name}")
    print(f"{'='*60}")
    
    parquet_file = pq.ParquetFile(parquet_path)
    print("Reading parquet in streaming batches...")
    processed_samples = []
    failed_count = 0
    part_idx = 0
    total_processed = 0

    # Choose only the columns we need to reduce memory usage (if present)
    desired_cols = ['audio', 'metadata', 'class', 'transcription', 'sentence']
    try:
        available_cols = [c for c in parquet_file.schema_arrow.names if c in desired_cols]
    except Exception:
        available_cols = None

    # Adaptive batch-size loop: try reading with CONFIG batch size, and on memory errors retry with smaller sizes
    batch_size = int(CONFIG.get('stream_batch_size', 128))
    min_batch_size = 1

    while True:
        try:
            # recreate parquet_file iterator each attempt to ensure a fresh read
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=available_cols):
                try:
                    df_batch = batch.to_pandas()
                except Exception as e:
                    print(f"Warning: could not convert batch to pandas: {e}")
                    continue

                # Filter out rows without audio
                if 'audio' not in df_batch.columns:
                    # free and continue
                    del df_batch
                    gc.collect()
                    continue

                df_batch = df_batch[df_batch['audio'].apply(lambda x: x is not None and x.get('array') is not None)]

                for _, row in df_batch.iterrows():
                    result = process_single_sample(row)
                    if result is not None:
                        processed_samples.append(result)
                        total_processed += 1
                    else:
                        failed_count += 1

                    # If we've buffered enough samples, save to disk and free memory
                    if len(processed_samples) >= CONFIG['save_batch_size']:
                        part_idx += 1
                        output_path = os.path.join(CONFIG['output_dir'], f'processed_{chunk_name}_part_{part_idx:04d}')
                        print(f"Saving part {part_idx} ({len(processed_samples)} samples) to {output_path}")
                        ds = Dataset.from_list(processed_samples)
                        ds = ds.cast_column("audio", Audio(sampling_rate=CONFIG['target_sampling_rate']))
                        ds.save_to_disk(output_path)
                        # clear buffers and run GC
                        processed_samples = []
                        ds = None
                        gc.collect()

                # free the batch dataframe to release memory
                del df_batch
                gc.collect()

            # Completed iteration successfully
            break

        except (MemoryError, RuntimeError) as e:
            msg = str(e)
            # Detect the pyarrow realloc OOM pattern and handle by reducing batch size
            if 'realloc' in msg or isinstance(e, MemoryError) or 'OutOfMemory' in msg or 'MemoryError' in msg:
                if batch_size <= min_batch_size:
                    print(f"Memory error persists even at batch_size={batch_size}. Aborting chunk {chunk_name}.")
                    return False
                new_batch = max(min_batch_size, batch_size // 2)
                print(f"Memory error reading parquet (batch_size={batch_size}). Reducing to {new_batch} and retrying...")
                batch_size = new_batch
                # reset partial state and try again from file start to avoid partial/duplicate reads
                processed_samples = []
                failed_count = 0
                part_idx = 0
                total_processed = 0
                gc.collect()
                continue
            else:
                print(f"Unhandled error while reading parquet: {e}")
                return False
    
    # Save any remaining processed samples
    if len(processed_samples) > 0:
        part_idx += 1
        output_path = os.path.join(CONFIG['output_dir'], f'processed_{chunk_name}_part_{part_idx:04d}')
        print(f"Saving final part {part_idx} ({len(processed_samples)} samples) to {output_path}")
        ds = Dataset.from_list(processed_samples)
        ds = ds.cast_column("audio", Audio(sampling_rate=CONFIG['target_sampling_rate']))
        ds.save_to_disk(output_path)
        total_processed += len(processed_samples)
        processed_samples = []
        ds = None
        gc.collect()
    
    print(f"\nTotal processed: {total_processed} samples")
    print(f"Failed: {failed_count} samples")
    print(f"Saved {part_idx} part(s) to: {CONFIG['output_dir']}")
    
    return True


# %%
def main():
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("Speech Dysfluency Detection - Preprocessing Pipeline")
    print("="*60)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted([
        f for f in os.listdir(CONFIG['input_dir']) 
        if f.endswith('.parquet') and f.startswith('chunk-')
    ])
    
    if len(parquet_files) == 0:
        print(f"ERROR: No parquet files found in {CONFIG['input_dir']}")
        print("Expected files: chunk-00001.parquet, chunk-00002.parquet, etc.")
        return
    
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  - {f}")
    
    # Process each chunk
    for parquet_file in parquet_files:
        chunk_name = parquet_file.replace('.parquet', '')
        parquet_path = os.path.join(CONFIG['input_dir'], parquet_file)
        
        try:
            dataset = process_parquet_chunk(parquet_path, chunk_name)
        except Exception as e:
            print(f"\nERROR processing {chunk_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"Processed datasets saved in: {CONFIG['output_dir']}")
    print("\nNext steps:")
    print("1. Load processed datasets using: Dataset.load_from_disk('processed_data/processed_chunk-00001')")
    print("2. Use these datasets for model training")

# %%
if __name__ == "__main__":
    main()


