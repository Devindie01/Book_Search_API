import pandas as pd
import numpy as np
import mlflow
import mlflow.sentence_transformers
from sentence_transformers import SentenceTransformer
import os

def train():
    # 1. Load Preprocessed Data
    processed_path = 'data/processed_books.pkl'
    if not os.path.exists(processed_path):
        print("Error: Processed data not found. Run preprocess.py first.")
        return
    
    df = pd.read_pickle(processed_path)

    # 2. Initialize Model
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    # 3. MLflow Experiment
    mlflow.set_experiment("Book_Semantic_Search")
    
    with mlflow.start_run(run_name="Initial_Embedding_Run"):
        print(f"Encoding {len(df)} books using {model_name}...")
        
        # Create vectors
        embeddings = model.encode(df['rich_text'].tolist(), show_progress_bar=True)
        
        # Log Parameters and the Model to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_records", len(df))
        mlflow.sentence_transformers.log_model(model, artifact_path="book_encoder")
        
        # Save artifacts for the FastAPI app
        os.makedirs("models", exist_ok=True)
        np.save("models/embeddings.npy", embeddings)
        
        # Save a slim version of metadata for the API to return results
        df[['book_id', 'title', 'authors', 'average_rating', 'image_url']].to_pickle("models/metadata.pkl")
        
        print("Success! Model and embeddings saved in /models and logged to MLflow.")

if __name__ == "__main__":
    train()