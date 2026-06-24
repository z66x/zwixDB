import os
import json
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
from core.embedder import TextEmbedder

def main():
    print("SYSTEM BOOT: Initializing Blockbuster Cinematic Ingestion Pipeline...")
    print("=" * 70)
    
    # 1. Fetching Data
    print("Establishing connection to Hugging Face registry...")
    try:
        movie_dataset = load_dataset("Pablinho/movies-dataset", split="train")
        print("Connection secure. Downloading database...")
    except Exception as e:
        print(f"FATAL ERROR: Failed to connect to registry. {e}")
        return

    # 2. Strict Schema Application
    print("Transforming to Pandas DataFrame...")
    df = movie_dataset.to_pandas()
    
    print(f"Detected columns in matrix: {df.columns.tolist()}")
    
    # Drop rows missing critical routing information using EXACT schema keys
    df = df.dropna(subset=['Title', 'Overview'])
    
    # Sort descending by the Popularity algorithm and slice the top 3000
    df = df.sort_values(by='Popularity', ascending=False).head(3000)
    
    # 3. Formatting the Corpus
    corpus = []
    metadata = []
    
    print(f"Extracting and enriching the top {len(df)} blockbuster profiles...")
    df = df.reset_index(drop=True)
    
    for idx, row in df.iterrows():
        title = str(row.get('Title', 'Unknown Title'))
        
        # Safely extract and slice the year from Release_Date (e.g., '1994-09-23' -> '1994')
        date_val = str(row.get('Release_Date', '0000'))
        year = date_val[:4] if date_val and date_val != 'nan' else '0000'
        
        overview = str(row.get('Overview', ''))
        genre = str(row.get('Genre', '')).replace('nan', '')
        poster_url = str(row.get('Poster_Url', '')).replace('nan', '')
        
        # The Enriched Semantic Fingerprint
        semantic_string = (
            f"Title: {title} ({year}). "
            f"Genre: {genre}. "
            f"Plot: {overview}"
        )
        
        corpus.append(semantic_string)
        
        # We map Genre to "genres" and leave "director" empty so api/main.py doesn't crash
        metadata.append({
            "id": idx,
            "title": title,
            "year": int(year) if year.isdigit() else 0,
            "genres": genre,
            "director": "", 
            "overview": overview,
            "poster_url": poster_url,
            "semantic_string": semantic_string
        })

    print(f"Data extraction complete. {len(corpus)} highly popular profiles formatted.")

    # 4. Vectorization
    print("\nBooting TextEmbedder to calculate multi-dimensional feature weights...")
    embedder = TextEmbedder()
    
    print("Executing batch vectorization...")
    t0 = time.time()
    embeddings = embedder.encode(corpus, batch_size=128, show_progress_bar=True)
    compilation_time = time.time() - t0
    
    print(f"Matrix compiled in {compilation_time:.2f} seconds.")
    print(f"Final Matrix Dimensionality: {embeddings.shape}")

    # 5. Storage Serialization
    print("\nCommitting processed assets to secure local storage...")
    os.makedirs("data/processed", exist_ok=True)
    
    matrix_path = "data/processed/tmdb_embeddings.npy"
    meta_path = "data/processed/tmdb_metadata.json"
    
    np.save(matrix_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=2)
        
    print(f" -> Matrix Weights written to: {matrix_path}")
    print(f" -> Catalog Metadata written to: {meta_path}")
    
    print("=" * 70)
    print("Data ingestion pipeline terminated successfully.")

if __name__ == "__main__":
    main()