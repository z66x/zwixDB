import numpy as np

class BruteForceEngine:
    def __init__(self, embeddings_matrix, metadata_list=None):
        """
        Initializes the exact mathematical baseline engine.
        Assumes the embeddings_matrix is pre-normalized to unit length.
        """
        self.embeddings = embeddings_matrix
        self.metadata = metadata_list if metadata_list is not None else []
        
        # Pre-extract years into a NumPy array for lightning-fast vectorized filtering
        if self.metadata:
            self.years_array = np.array([meta.get('year', 0) for meta in self.metadata])
        else:
            self.years_array = np.array([])

    def knn_search(self, query_vec, k=3, filter_dict=None):
        """
        Executes an exact O(N) linear scan using pure NumPy dot products.
        Returns data in the exact same format as ZwixHNSW: [(node_id, distance), ...]
        """
        # --- FIX 3: Bulletproof Normalization Guard ---
        # Enforce query vector is strictly unit length to prevent silent math failures
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        # 1. Calculate exact Cosine Similarities for the entire database instantly
        similarities = np.dot(self.embeddings, query_vec)
        
        # 2. Vectorized Metadata Masking
        if filter_dict and "min_year" in filter_dict and len(self.years_array) > 0:
            min_year = filter_dict["min_year"]
            valid_mask = self.years_array >= min_year
            # Apply mask: force invalid indices to -2.0 (lowest possible cosine sim is -1.0)
            similarities = np.where(valid_mask, similarities, -2.0)

        # --- FIX 1: O(N) Top-K Extraction ---
        # argpartition safely isolates the top K elements in O(N) time without fully sorting
        safe_k = min(k, len(similarities))
        top_k_unsorted_indices = np.argpartition(similarities, -safe_k)[-safe_k:]
        
        # Now we only run O(K log K) sorting on the tiny isolated subset
        top_indices = top_k_unsorted_indices[np.argsort(similarities[top_k_unsorted_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            # --- FIX 2: Explicit Guard against strict filters ---
            # If the filter was so strict that less than k valid results exist, 
            # we will inevitably hit our -2.0 mask. Break out to avoid returning garbage.
            if similarities[idx] == -2.0:
                break
            
            dist = 1.0 - similarities[idx]
            results.append((int(idx), dist))
            
        return results