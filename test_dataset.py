# check_data.py
import torch
import glob
import os

files = glob.glob("dataset/processed/sample_*.pt")
print(f"Checking {len(files)} files...")

stutter_counts = {1: 0, 2: 0, 3: 0, 4: 0}
files_with_stutter = 0

for f in files:
    data = torch.load(f)
    labels = data['labels']
    
    # Check if any label is not 0 (Fluent)
    unique_labels = torch.unique(labels)
    if any(l > 0 for l in unique_labels):
        files_with_stutter += 1
        for l in unique_labels:
            if l.item() > 0:
                stutter_counts[l.item()] += 1

print(f"Files containing valid stutters: {files_with_stutter}/{len(files)}")
print(f"Stutter class distribution (Files present in): {stutter_counts}")


# import pandas as pd
# import glob
# import os

# # Find the first parquet file
# raw_dir = 'dataset/raw'
# files = glob.glob(os.path.join(raw_dir, "*.parquet"))

# if not files:
#     print(f"No parquet files found in {raw_dir}")
# else:
#     print(f"Inspecting file: {files[0]}")
    
#     # Load the dataframe
#     df = pd.read_parquet(files[0])
    
#     print("\n--- COLUMN NAMES ---")
#     print(df.columns.tolist())
    
#     print("\n--- FIRST 5 ROWS ---")
#     # We display transpose so it's easier to read
#     print(df.head().T)

#     # Check unique values in likely label columns to see if they match your CONFIG keys
#     # I suspect the issue is capitalization (e.g., 'Block' vs 'block')
#     likely_columns = [c for c in df.columns if 'type' in c.lower() or 'label' in c.lower() or 'stutter' in c.lower()]
    
#     for col in likely_columns:
#         print(f"\n--- UNIQUE VALUES IN '{col}' ---")
#         print(df[col].unique())