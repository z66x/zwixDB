import numpy as np
import torch
import logging
from sentence_transformers import SentenceTransformer

# Configure a clean logger for system events
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the embedding model with automatic hardware acceleration detection.
        """
        # Automatically detect the best available hardware
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        logging.info(f"Booting TextEmbedder using device: {self.device.upper()}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        
        logging.info(f"Model '{model_name}' loaded. Vector dimension: {self.vector_dimension}")

    def encode(self, texts, batch_size=128, show_progress_bar=True):
        """
        Encodes a list of strings into a normalized 2D NumPy matrix.
        Optimized for bulk dataset ingestion.
        """
        if not texts:
            return np.array([])
            
        # 1. Generate raw embeddings via the transformer model
        raw_embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar
        )
        
        # 2. Centralized Normalization Guard
        # Calculate the L2 norm for each row and divide to force length to 1.0
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        normalized_matrix = raw_embeddings / (norms + 1e-9)
        
        return normalized_matrix

    def encode_query(self, query_text):
        """
        Encodes a single query string into a normalized 1D NumPy array.
        Optimized for real-time user searches.
        """
        # Encode returns a 2D array, so we slice [0] to get the flat vector
        raw_vector = self.model.encode([query_text])[0]
        
        # Enforce unit normalization for the single query
        normalized_vector = raw_vector / (np.linalg.norm(raw_vector) + 1e-9)
        
        return normalized_vector