import os
import time
import json
import pickle
import numpy as np
from datasets import load_dataset
from core.hnsw import ZwixHNSW
from core.brute_force import BruteForceEngine
from core.embedder import TextEmbedder

def calculate_ann_recall(exact_ids, pred_ids):
    """Calculates algorithmic accuracy: how many true mathematical nearest neighbors HNSW found."""
    intersection = set(exact_ids).intersection(set(pred_ids))
    return len(intersection) / len(exact_ids) if exact_ids else 0.0

def main():
    print("SYSTEM BOOT: Initializing multi-phase research benchmark suite...")
    print("=" * 70)
    
    # 1. Load Core Assets
    print("Mounting SciDocs matrix and metadata from secure storage...")
    full_embeddings = np.load("data/processed/scidocs_embeddings.npy")
    with open("data/processed/scidocs_metadata.json", "r") as f:
        full_metadata = json.load(f)["metadata"]

    print("Fetching authentic BEIR query corpus for statistical aggregation...")
    queries_dataset = load_dataset("BeIR/scidocs", "queries")["queries"]
    test_queries = [q["text"] for q in queries_dataset if len(q["text"].strip()) > 10][:200]

    # 2. Pre-Encode Queries (The Wall-Clock Optimization)
    print("\nWaking up transformer neural network...")
    embedder = TextEmbedder()
    
    print("Pre-encoding batch of 200 evaluation queries to isolate search metrics...")
    query_vectors = embedder.encode(test_queries, batch_size=64, show_progress_bar=False)
    print("Query vectorization complete.")

    # --- WARMUP ROUTINE ---
    print("Executing hardware warmup sequence to prime JIT compiler and CPU caches...")
    dummy_engine = BruteForceEngine(embeddings_matrix=full_embeddings[:100], metadata_list=full_metadata[:100])
    dummy_engine.knn_search(query_vectors[0], k=10)
    print("Warmup complete. Hardware ready.")

    benchmark_report = {
        "metadata": {
            "queries_executed": len(test_queries),
            "k_neighbors": 10
        },
        "phase_1_scale_sweep": {},
        "phase_2_ef_sweep": {}
    }

    os.makedirs("benchmarks/results", exist_ok=True)
    k = 10

    # =========================================================================
    # PHASE 1: SCALE SWEEP (O(N) vs O(log N))
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: SCALE SWEEP EVALUATION (Fixed ef_search=50)")
    print("=" * 70)

    scale_steps = [1000, 5000, 10000, 25000]
    if len(full_embeddings) >= 50000:
        scale_steps.append(50000)

    # We will keep a reference to the largest engine built for Phase 2
    largest_hnsw_engine = None
    largest_size = 0

    for size in scale_steps:
        actual_size = min(size, len(full_embeddings))
        print(f"\n>>> CONFIGURING TIER: {actual_size} DOCUMENTS <<<")
        
        subset_embeddings = full_embeddings[:actual_size]
        subset_metadata = full_metadata[:actual_size]

        bf_engine = BruteForceEngine(embeddings_matrix=subset_embeddings, metadata_list=subset_metadata)
        hnsw_engine = ZwixHNSW(M=16, ef_construction=200, ef_search=50, metadata_store=subset_metadata)
        
        print(f"Stitching HNSW Graph Topology for N={actual_size}...")
        t0_build = time.time()
        for idx, vec in enumerate(subset_embeddings):
            hnsw_engine.insert(node_id=idx, vector=vec)
        build_time = time.time() - t0_build
        print(f"Graph compilation complete in {build_time:.2f} seconds.")

        # Checkpointing
        index_path = f"benchmarks/results/hnsw_index_{actual_size}.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(hnsw_engine, f)

        # Store for Phase 2
        largest_hnsw_engine = hnsw_engine
        largest_size = actual_size

        print("Executing batch inference...")
        total_recall = 0.0
        bf_total_time = 0.0
        hnsw_total_time = 0.0

        # Iterating over the pre-computed query vectors
        for query_vec in query_vectors:
            # Baseline execution
            t0 = time.time()
            bf_results = bf_engine.knn_search(query_vec, k=k)
            bf_total_time += (time.time() - t0)
            
            # Graph execution
            t0 = time.time()
            hnsw_results = hnsw_engine.knn_search(query_vec, k=k)
            hnsw_total_time += (time.time() - t0)
            
            bf_ids = [res[0] for res in bf_results]
            hnsw_ids = [res[0] for res in hnsw_results]
            total_recall += calculate_ann_recall(bf_ids, hnsw_ids)

        avg_recall = total_recall / len(query_vectors)
        avg_bf_ms = (bf_total_time / len(query_vectors)) * 1000
        avg_hnsw_ms = (hnsw_total_time / len(query_vectors)) * 1000
        speedup = avg_bf_ms / avg_hnsw_ms

        print(f"Tier {actual_size} Results:")
        print(f" -> ANN Recall@{k}: {avg_recall*100:.2f}%")
        print(f" -> Brute Force Latency: {avg_bf_ms:.3f} ms")
        print(f" -> HNSW Latency: {avg_hnsw_ms:.3f} ms")
        print(f" -> Speed Multiplier: {speedup:.2f}x")

        benchmark_report["phase_1_scale_sweep"][str(actual_size)] = {
            "build_time_seconds": build_time,
            "avg_ann_recall": avg_recall,
            "avg_brute_force_ms": avg_bf_ms,
            "avg_hnsw_ms": avg_hnsw_ms,
            "speedup_factor": speedup
        }

    # =========================================================================
    # PHASE 2: PARAMETER SWEEP (Recall vs Latency Tuning)
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: HYPERPARAMETER TUNING (Fixed N={largest_size})")
    print("=" * 70)
    print("Sweeping ef_search to map the Pareto frontier...")

    ef_sweep_values = [10, 20, 50, 100, 200, 400]
    
    # We only need the baseline results once for this exact subset size to calculate recall
    subset_embeddings = full_embeddings[:largest_size]
    subset_metadata = full_metadata[:largest_size]
    bf_engine = BruteForceEngine(embeddings_matrix=subset_embeddings, metadata_list=subset_metadata)
    
    baseline_results = []
    for query_vec in query_vectors:
        baseline_results.append([res[0] for res in bf_engine.knn_search(query_vec, k=k)])

    for ef in ef_sweep_values:
        largest_hnsw_engine.ef_search = ef
        
        total_recall = 0.0
        hnsw_total_time = 0.0
        
        for idx, query_vec in enumerate(query_vectors):
            t0 = time.time()
            hnsw_results = largest_hnsw_engine.knn_search(query_vec, k=k)
            hnsw_total_time += (time.time() - t0)
            
            hnsw_ids = [res[0] for res in hnsw_results]
            total_recall += calculate_ann_recall(baseline_results[idx], hnsw_ids)
            
        avg_recall = total_recall / len(query_vectors)
        avg_hnsw_ms = (hnsw_total_time / len(query_vectors)) * 1000
        
        print(f"ef_search={ef:<3} | Recall@{k}: {avg_recall*100:5.2f}% | Latency: {avg_hnsw_ms:6.3f} ms")
        
        benchmark_report["phase_2_ef_sweep"][str(ef)] = {
            "avg_ann_recall": avg_recall,
            "avg_hnsw_ms": avg_hnsw_ms
        }

    # 4. Final Save
    report_path = "benchmarks/results/comprehensive_eval.json"
    with open(report_path, "w") as f:
        json.dump(benchmark_report, f, indent=4)
    print("\n" + "=" * 70)
    print(f"Research evaluation complete. Full diagnostic report written to '{report_path}'.")
    print("System terminating.")

if __name__ == "__main__":
    main()