import time
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.hnsw import ZwixHNSW
from core.brute_force import BruteForceEngine
from core.embedder import TextEmbedder
from api.models import QueryRequest, SearchResponse, SearchResult

# Global state pointers
embedder = None
hnsw_engine = None
bf_engine = None
metadata_store = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, hnsw_engine, bf_engine, metadata_store
    
    print("\n[INIT] Waking up neural network models...")
    embedder = TextEmbedder()
    
    print("[INIT] Loading production cinematic matrix from disk...")
    try:
        embeddings = np.load("data/processed/tmdb_embeddings.npy")
        with open("data/processed/tmdb_metadata.json", "r") as f:
            metadata_store = json.load(f)["metadata"]
    except FileNotFoundError:
        print("[FATAL] Production data not found. Run prepare_tmdb.py first.")
        raise
        
    print("[INIT] Allocating exact mathematical baseline (Brute Force Engine)...")
    bf_engine = BruteForceEngine(embeddings_matrix=embeddings, metadata_list=metadata_store)
        
    print("[INIT] Stitching HNSW topology into active memory...")
    hnsw_engine = ZwixHNSW(M=16, ef_search=50, metadata_store=metadata_store)
    
    for idx, vec in enumerate(embeddings):
        hnsw_engine.insert(node_id=idx, vector=vec)
        
    print(f"[SUCCESS] Server online. Graph loaded with {len(embeddings)} records.")
    yield
    print("\n[SHUTDOWN] Purging memory buffers...")

app = FastAPI(title="Semantic Cinema API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    """Frontend polling endpoint to verify system readiness."""
    return {
        "status": "online",
        "graph_loaded": hnsw_engine is not None,
        "baseline_loaded": bf_engine is not None,
        "records": len(metadata_store)
    }

@app.post("/search/graph", response_model=SearchResponse)
async def search_graph(request: QueryRequest):
    """Executes high-speed approximate retrieval using the multi-layer skip graph."""
    if hnsw_engine is None or embedder is None:
        raise HTTPException(status_code=503, detail="System engines are still booting.")
        
    t0 = time.time()
    query_vector = embedder.encode_query(request.query)
    
    filter_dict = {}
    if request.filters and request.filters.min_year:
        filter_dict["min_year"] = request.filters.min_year
        
    raw_results = hnsw_engine.knn_search(query_vector, k=request.k, filter_dict=filter_dict)
    
    formatted_results = []
    for node_id, dist in raw_results:
        meta = metadata_store[node_id]
        formatted_results.append(SearchResult(
            movie_id=meta.get("id", node_id),
            title=meta.get("title", "Unknown"),
            year=meta.get("year", 0),
            genres=meta.get("genres", ""),
            director=meta.get("director", ""),
            overview=meta.get("overview", ""),
            distance=float(dist)
        ))
        
    latency = (time.time() - t0) * 1000
    
    return SearchResponse(
        results=formatted_results,
        latency_ms=latency,
        engine="Custom HNSW Graph"
    )

@app.post("/search/brute", response_model=SearchResponse)
async def search_brute(request: QueryRequest):
    """Executes an exact O(N) linear array scan for baseline comparison."""
    if bf_engine is None or embedder is None:
        raise HTTPException(status_code=503, detail="System engines are still booting.")
        
    t0 = time.time()
    query_vector = embedder.encode_query(request.query)
    
    filter_dict = {}
    if request.filters and request.filters.min_year:
        filter_dict["min_year"] = request.filters.min_year
        
    raw_results = bf_engine.knn_search(query_vector, k=request.k, filter_dict=filter_dict)
    
    formatted_results = []
    for node_id, dist in raw_results:
        meta = metadata_store[node_id]
        formatted_results.append(SearchResult(
            movie_id=meta.get("id", node_id),
            title=meta.get("title", "Unknown"),
            year=meta.get("year", 0),
            genres=meta.get("genres", ""),
            director=meta.get("director", ""),
            overview=meta.get("overview", ""),
            distance=float(dist)
        ))
        
    latency = (time.time() - t0) * 1000
    
    return SearchResponse(
        results=formatted_results,
        latency_ms=latency,
        engine="Exact Linear Scan (O(N))"
    )