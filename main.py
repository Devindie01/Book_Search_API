from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
import numpy as np
from function import returnSearchResultIndexes

# 1. Configuration & Loading
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Load your book index (saved as Parquet for Polars efficiency)
# This file should contain book metadata + embedding columns
df = pl.scan_parquet("data/book-index.parquet")

# Manhattan distance is what you specified in your architecture
dist = DistanceMetric.get_metric("manhattan")

app = FastAPI(title="Book Semantic Search API")


@app.get("/")
def health_check():
    return {"status": "active", "dataset": "Books NLP"}


@app.get("/search")
def search(query: str):
    # 1. Get the indices of the top matches
    idx_result = returnSearchResultIndexes(query, df, model, dist)

    # 2. Select metadata columns and execute the query (collect)
    # We collect the metadata into a standard DataFrame first
    metadata_df = df.select(
        ["title", "authors", "average_rating", "image_url"]
    ).collect()

    # 3. Use the indices to filter the DataFrame and convert to a list of dicts
    # .to_dicts() automatically groups data by row: [{book1}, {book2}, ...]
    results = metadata_df[idx_result].to_dicts()

    return {"query": query, "results": results}
