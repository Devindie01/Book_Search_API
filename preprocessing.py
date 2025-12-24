import pandas as pd
import os


def preprocess_books(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Clean missing values
    # It's important to fillna('') so the concatenation doesn't result in 'NaN' strings
    df["title"] = df["title"].fillna("Unknown Title")
    df["authors"] = df["authors"].fillna("Unknown Author")
    df["description"] = df["description"].fillna("")

    # 2. Combine text columns into a single field for the encoder
    print("Combining text columns for semantic encoding...")
    df["rich_text"] = (
        "Title: "
        + df["title"]
        + " | Author: "
        + df["authors"]
        + " | Description: "
        + df["description"]
    )

    # 3. Save the processed data
    # We save as a pickle to preserve data types and the new column
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_books("data/book.csv", "data/processed_books.pkl")
