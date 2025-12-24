import pandas as pd
import numpy as np
import polars as pl
import os

def create_index():
    # 1. Load the metadata and embeddings
    metadata = pd.read_pickle("models/metadata.pkl")
    embeddings = np.load("models/embeddings.npy")
    
    # 2. Convert embeddings to a Polars DataFrame
    # We name columns 'column_0', 'column_1', etc. to match the API's regex
    emb_df = pl.from_numpy(
        embeddings, 
        schema=[f"column_{i}" for i in range(embeddings.shape[1])]
    )
    
    # 3. Convert metadata to Polars
    meta_df = pl.from_pandas(metadata)
    
    # 4. Horizontal Concatenation (Glue them side-by-side)
    final_df = pl.concat([meta_df, emb_df], how="horizontal")
    
    # 5. Save as Parquet for the API
    os.makedirs("data", exist_ok=True)
    final_df.write_parquet("data/book-index.parquet")
    print("Successfully created data/book-index.parquet")

if __name__ == "__main__":
    create_index()