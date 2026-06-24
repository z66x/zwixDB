import numpy as np
import random
import heapq

class ZwixHNSW:
    def __init__(self, M=16, ef_construction=200, ef_search=50, metadata_store=None):
        self.M = M
        self.M_max = M 
        self.M_max0 = 2 * M 
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / np.log(M)
        
        self.vectors = {}      # ID -> Normalized Vector
        self.graph = {}        # layer -> { node_id: set(neighbors) }
        self.metadata_store = metadata_store if metadata_store is not None else {}
        
        self.max_layer = -1    
        self.enter_node = None 

    def _get_random_layer(self):
        r = random.random()
        if r == 0: r = 1e-9
        return int(-np.log(r) * self.mL)

    def _distance(self, vec1, vec2):
        return 1.0 - np.dot(vec1, vec2)
    
    def _search_layer(self, query_vec, enter_nodes, ef, layer_idx, filter_dict=None):
        """
        Executes an advanced best-first beam search across a single layer.
        Patched for strict initialization filtering and safe bounded early exits.
        """
        visited = set(enter_nodes)
        v_candidates = []
        w_results = []
        
        # Initialize queues with starting entry points
        for ep in enter_nodes:
            dist = self._distance(query_vec, self.vectors[ep])
            heapq.heappush(v_candidates, (dist, ep))
            
            # --- FIX 1: Apply filter check during initialization ---
            if filter_dict and "min_year" in filter_dict:
                if self.metadata_store.get(ep, {}).get('year', 0) < filter_dict["min_year"]:
                    continue # Do not add to the valid results pool, but leave in candidates for routing
                    
            heapq.heappush(w_results, (-dist, ep))
            
        while len(v_candidates) > 0:
            curr_dist, curr_node = heapq.heappop(v_candidates)
            
            # --- FIX 2: Bounded Early-Exit Condition ---
            # Only break if we have enough elements AND the candidate is worse than our worst result
            if len(w_results) >= ef:
                farthest_result_dist = -w_results[0][0]
                if curr_dist > farthest_result_dist:
                    break
                
            neighbors = self.graph[layer_idx].get(curr_node, set())
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                # Apply metadata criteria gate during exploration
                if filter_dict and "min_year" in filter_dict:
                    if self.metadata_store.get(neighbor, {}).get('year', 0) < filter_dict["min_year"]:
                        continue
                        
                dist = self._distance(query_vec, self.vectors[neighbor])
                
                # Calculate current threshold safely
                farthest_result_dist = -w_results[0][0] if len(w_results) > 0 else float('inf')
                
                if dist < farthest_result_dist or len(w_results) < ef:
                    heapq.heappush(v_candidates, (dist, neighbor))
                    heapq.heappush(w_results, (-dist, neighbor))
                    
                    if len(w_results) > ef:
                        heapq.heappop(w_results)
                        
        return {node_id for (_, node_id) in w_results}

    def insert(self, node_id, vector, metadata=None):
        self.vectors[node_id] = vector
        if metadata:
            self.metadata_store[node_id] = metadata
            
        if self.enter_node is None:
            self.enter_node = node_id
            self.max_layer = 0
            self.graph[0] = {node_id: set()}
            return

        new_node_max_layer = self._get_random_layer()
        
        # Phase 1: Pure greedy top-down descent down to the entry layer
        curr_eps = {self.enter_node}
        for layer_idx in range(self.max_layer, new_node_max_layer, -1):
            # Pass ef=1 for pure greedy routing across upper structural horizons
            curr_eps = self._search_layer(vector, curr_eps, ef=1, layer_idx=layer_idx)
            
        # Phase 2: High-accuracy beam search insertion using ef_construction
        for layer_idx in range(min(self.max_layer, new_node_max_layer), -1, -1):
            if layer_idx not in self.graph:
                self.graph[layer_idx] = {}
                
            # Broad search using ef_construction width
            candidates = self._search_layer(vector, curr_eps, ef=self.ef_construction, layer_idx=layer_idx)
            
            if node_id not in self.graph[layer_idx]:
                self.graph[layer_idx][node_id] = set()
                
            # Select nearest M connections from candidates list pool
            # (We will sort by raw distance here, paving the way for the diversity heuristic next)
            sorted_candidates = list(candidates)
            sorted_candidates.sort(key=lambda c: self._distance(vector, self.vectors[c]))
            chosen_neighbors = sorted_candidates[:self.M]
            
            # Establish bi-directional link maps
            for neighbor in chosen_neighbors:
                self.graph[layer_idx][node_id].add(neighbor)
                if neighbor not in self.graph[layer_idx]:
                    self.graph[layer_idx][neighbor] = set()
                self.graph[layer_idx][neighbor].add(node_id)
                
                # Prune old neighbor if it exceeds capacity parameters
                current_cap = self.M_max0 if layer_idx == 0 else self.M_max
                if len(self.graph[layer_idx][neighbor]) > current_cap:
                    n_list = list(self.graph[layer_idx][neighbor])
                    n_list.sort(key=lambda x: self._distance(self.vectors[neighbor], self.vectors[x]))
                    self.graph[layer_idx][neighbor] = set(n_list[:current_cap])
            
            # The current candidate pool serves as entry points for the next floor down
            curr_eps = candidates

        # Handle global pointer updates if node shoots above roof
        if new_node_max_layer > self.max_layer:
            for layer_idx in range(self.max_layer + 1, new_node_max_layer + 1):
                self.graph[layer_idx] = {node_id: set()}
            self.max_layer = new_node_max_layer
            self.enter_node = node_id

    def knn_search(self, query_vec, k=3, filter_dict=None):
        if self.enter_node is None:
            return []
            
        curr_eps = {self.enter_node}
        
        # Traverse upper levels using fast greedy checks (ef=1)
        for layer_idx in range(self.max_layer, 0, -1):
            curr_eps = self._search_layer(query_vec, curr_eps, ef=1, layer_idx=layer_idx)
            
        # At street level (Layer 0), open the beam width to our full ef_search constraint
        final_candidates = self._search_layer(query_vec, curr_eps, ef=self.ef_search, layer_idx=0, filter_dict=filter_dict)
        
        # Sort and return the top k items
        results = []
        for node_id in final_candidates:
            dist = self._distance(query_vec, self.vectors[node_id])
            results.append((node_id, dist))
            
        results.sort(key=lambda x: x[1])
        return results[:k]