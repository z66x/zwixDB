import os
import json
import time
import numpy as np
from datasets import load_dataset
from core.embedder import TextEmbedder

def main():
    print("SYSTEM BOOT: Initializing Academic Benchmark Data Pipeline...")
    print("=" * 70)
    
    # 1. Fetching Data
    print("Establishing connection to BEIR/SciDocs registry...")
    try:
        # We use the Hugging Face mirror of BEIR for seamless dependency management
        dataset = load_dataset("BeIR/scidocs", "corpus", split="corpus")
        print("Connection secure. Downloading academic corpus...")
    except Exception as e:
        print(f"FATAL ERROR: Failed to connect to registry. {e}")
        return

    # 2. Formatting the Corpus
    corpus = []
    metadata = []
    
    print("Extracting and formatting academic documents...")
    
    for idx, doc in enumerate(dataset):
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        
        # Filter out completely empty documents
        if not text:
            continue
            
        # The Semantic Fingerprint
        semantic_string = f"Document Title: {title}. Abstract: {text}"
        
        corpus.append(semantic_string)
        metadata.append({
            "id": idx,
            "title": title,
            "text": text
        })

    actual_count = len(corpus)
    print(f"Data extraction complete. {actual_count} valid academic papers formatted.")

    # 3. Vectorization
    print("\nBooting TextEmbedder to calculate mathematical feature weights...")
    embedder = TextEmbedder()
    
    print("Executing batch vectorization (This will take a few minutes)...")
    t0 = time.time()
    embeddings = embedder.encode(corpus, batch_size=128, show_progress_bar=True)
    compilation_time = time.time() - t0
    
    print(f"Matrix compiled in {compilation_time:.2f} seconds.")
    print(f"Final Matrix Dimensionality: {embeddings.shape}")

    # 4. Storage Serialization
    print("\nCommitting benchmark assets to secure local storage...")
    os.makedirs("data/processed", exist_ok=True)
    
    matrix_path = "data/processed/scidocs_embeddings.npy"
    meta_path = "data/processed/scidocs_metadata.json"
    
    np.save(matrix_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=2)
        
    print(f" -> Matrix Weights written to: {matrix_path}")
    print(f" -> Catalog Metadata written to: {meta_path}")
    
    print("=" * 70)
    print("Benchmark data ingestion complete. System ready for evaluation sweeps.")

if __name__ == "__main__":
    main()