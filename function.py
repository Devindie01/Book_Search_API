import numpy as np
import polars as pl
from sklearn.metrics import DistanceMetric


def returnSearchResultIndexes(
    query: str, df: pl.LazyFrame, model, dist: DistanceMetric
) -> np.ndarray:
    """
    Function to return indexes of top search results for books
    """
    # 1. Embed the user query (convert text to vector)
    query_embedding = model.encode(query).reshape(1, -1)

    # 2. Extract embeddings from Polars LazyFrame
    # We assume the embedding columns are at specific indices (e.g., 5 to 389)
    # This matches the 384 dimensions of 'all-MiniLM-L6-v2'
    embeddings_df = df.select(pl.col("^column_.*$")).collect()

    # 3. Compute distances
    # Manhattan distance: smaller value = more similar
    dist_arr = dist.pairwise(embeddings_df.to_numpy(), query_embedding)

    # 4. Search parameters
    top_k = 5

    # 5. Get top K closest matches (indices with smallest distance)
    # Using argpartition is faster than full sort for getting top K
    idx_result = np.argpartition(dist_arr.flatten(), top_k)[:top_k]

    # Sort those top K results by distance
    idx_sorted = idx_result[np.argsort(dist_arr.flatten()[idx_result])]

    return idx_sorted
