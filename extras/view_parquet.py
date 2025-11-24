import pyarrow.parquet as pq
import pandas as pd
import dask.dataframe as dd

file_path = r'E:\College\Final Y\Sem I\EDAI\stuttering\dataset\raw\chunk-00001.parquet'

# Open the Parquet file
try:
    parquet_file = pq.ParquetFile(file_path)

    # Print the schema (column names and types)
    print("--- 🔍 File Schema ---")
    print(parquet_file.schema)

    # Print more detailed metadata (row groups, creator, etc.)
    print("\n--- ℹ️ File Metadata ---")
    print(parquet_file.metadata)

    ###################################################
    
    # first_batch = next(parquet_file.iter_batches(batch_size=3))
    # df = first_batch.to_pandas()

    # # Save to CSV
    # df.to_csv("first3.csv", index=False)

    ################################################

    # Read only the metadata struct column
    table = pq.read_table(file_path, columns=["metadata"])

    # Convert struct to Python dictionaries
    metadata_list = table["metadata"].to_pylist()

    # Extract stutter_type from each row safely
    unique_types = set()

    for row in metadata_list:
        if row and "stutter_type" in row and row["stutter_type"] is not None:
            unique_types.add(row["stutter_type"])

    print(f"\n\n{unique_types}")

    ##################################################

    metadata = pq.read_metadata(file_path)
    print(f"\n\nRow count: {metadata.num_rows}")



except FileNotFoundError:
    print(f"Error: File not found at {file_path}")

