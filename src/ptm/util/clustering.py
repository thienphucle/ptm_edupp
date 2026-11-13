"""
Handles the consensus clustering of "keyinfo" instances using a multi-step approach
including semantic embedding, topic classification, temporal segregation, and recursive splitting.
"""
import json
import os
import re
import time
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field
import numpy as np
import umap
import hdbscan
from sklearn.cluster import KMeans

# Use relative imports for the new package structure
from . import config
from .datastructure import KeyInfoInstance
from .llm_handler import get_cluster_topic

# Initialize the embedding model once
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            print("[Clustering] Loading sentence transformer embedding model (lazy)...")
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            print("[Clustering] Embedding model loaded successfully.")
        except Exception as e:
            print(f"[Clustering] WARNING: Could not load embedding model. Clustering may fail. Error: {e}")
    return embedding_model

# --- HELPER FUNCTIONS (Moved from ccluster.py) ---

def create_semantic_fingerprints(instances: List[KeyInfoInstance], dataset_name: str) -> List[str]:
    """Creates descriptive strings focusing on WHAT, WHY, WHERE."""
    print("[Step 1a] Creating semantic fingerprints...")
    fingerprints = []
    weights = config.FEATURE_WEIGHTS.get(dataset_name, {"WHAT": 1, "HOW": 1, "WHERE": 1})
    
    for inst in instances:
        parts = []
        
        if inst.WHAT and inst.WHAT.strip():
            parts.extend([f"Action: {inst.WHAT}"] * weights.get("WHAT", 1))  #with weight applied
            
        if inst.HOW and inst.HOW.strip():
            parts.extend([f"Evidence: {inst.HOW}"] * weights.get("HOW", 1))
            
        if inst.WHERE and inst.WHERE.strip():
            parts.extend([f"Context: {inst.WHERE}"] * weights.get("WHERE", 1))
            
        fingerprint = ". ".join(parts)
        
        # Fallback if WHAT, HOW, WHERE are all empty
        if not fingerprint:
            fingerprint = f"Instance {inst.provenance}" 

        fingerprints.append(fingerprint)
    
    print(f"  > Created {len(fingerprints)} weighted fingerprints.")
    return fingerprints

def embed_fingerprints(fingerprints: List[str]) -> Optional[np.ndarray]:
    """Generates embeddings for a list of fingerprints."""
    print("[Step 1b] Generating embeddings from fingerprints...")
    model = get_embedding_model()
    if not model:
        print("  [ERROR] Embedding model not loaded.")
        return None
    if not fingerprints:
        print("  [ERROR] No fingerprints to embed.")
        return None
        
    try:
        embeddings = model.encode(fingerprints, show_progress_bar=False) # Use False for cleaner logs
        print(f"  > Generated embedding matrix of shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"  [ERROR] Failed to generate embeddings: {e}")
        return None

def run_umap(embeddings: np.ndarray, log_prefix: str = "  ") -> Optional[np.ndarray]:
    """Helper function to run UMAP reduction."""
    try:
        n_samples = embeddings.shape[0]
        
        # Handle edge case: If there are too few samples, skip UMAP and return original embeddings
        if n_samples < 5:
            if log_prefix == "  ":
                print(f"  [UMAP] Skipping reduction (too few samples: {n_samples}).")
            return embeddings

        # Ensure n_components and n_neighbors are within valid bounds
        safe_n_components = min(config.UMAP_N_COMPONENTS, n_samples - 2)
        safe_n_neighbors = min(config.UMAP_N_NEIGHBORS, n_samples - 1)
        safe_n_neighbors = max(2, safe_n_neighbors) # Ensure at least 2 neighbors for UMAP

        if log_prefix == "  ":
            print(f"  [UMAP] Reducing dimensionality (n_samples={n_samples}, n_components={safe_n_components}, n_neighbors={safe_n_neighbors})...")
            
        reducer = umap.UMAP(
            n_components=safe_n_components, 
            n_neighbors=safe_n_neighbors,
            min_dist=config.UMAP_MIN_DIST,
            metric='cosine',
            random_state=42,
            n_jobs=1 # Avoids potential conflicts in threaded environments
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        if log_prefix == "  ":
            print(f"    > Reduced embeddings shape: {reduced_embeddings.shape}")
        return reduced_embeddings
    except Exception as e:
        print(f"{log_prefix}[ERROR] UMAP failed: {e}")
        return None

def run_hdbscan_clustering(reduced_embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Runs HDBSCAN clustering."""
    print(f"  [HDBSCAN] Clustering with min_cluster_size={min_cluster_size}...")
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            allow_single_cluster=True,
            # core_dist_n_jobs=1 # May help with stability in some envs
        )
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_noise = np.count_nonzero(cluster_labels == -1)
        print(f"    > Found {num_clusters} semantic clusters and {num_noise} noise points.")
        return cluster_labels
    except Exception as e:
        print(f"    [ERROR] HDBSCAN failed: {e}")
        # Return all as noise if HDBSCAN fails
        return np.full(reduced_embeddings.shape[0], -1)


def group_by_label(cluster_labels: np.ndarray, instance_indices_map: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """Groups instance indices by cluster label, mapping back if needed."""
    groups = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        # We process noise points (label -1) as well now
        original_instance_index = instance_indices_map[i] if instance_indices_map else i
        groups[int(label)].append(original_instance_index) # Ensure label is int
    return dict(groups) # Convert back to dict

# --- Recursive Segregation Logic ---

def _recursive_segregate(
    instance_indices: List[int],
    current_key_prefix: str,
    instances: List[KeyInfoInstance],
    all_embeddings: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Recursively splits and classifies clusters until they are below MAX_CLUSTER_SIZE.
    Uses K-Means to force splits on large clusters.
    Relies on get_cluster_topic imported from llm_handler.
    """
    resolved_groups = {}

    # --- BASE CASE ---
    if len(instance_indices) <= config.MAX_CLUSTER_SIZE:
        print(f"  [Topic] Classifying topic for group '{current_key_prefix}' (size: {len(instance_indices)})...")
        cluster_instances = [instances[i] for i in instance_indices]
        # Calls the function imported from llm_handler
        topic = get_cluster_topic(cluster_instances)
        print(f"    > Topic classified as: '{topic}'")
        
        final_key = f"{current_key_prefix}_topic_{topic}"
        resolved_groups[final_key] = instance_indices
        return resolved_groups
    
    # --- RECURSIVE CASE ---
    print(f"  [Split] Group '{current_key_prefix}' is too large ({len(instance_indices)} instances). Splitting with K-Means...")

    # 1. Get embeddings for this group
    cluster_embeddings = all_embeddings[instance_indices]
    
    # 2. Run UMAP *again* on this subset
    reduced_cluster_embeddings = run_umap(cluster_embeddings, log_prefix="    ")
    if reduced_cluster_embeddings is None: # Handle UMAP failure
         print(f"    [WARN] UMAP failed for splitting '{current_key_prefix}'. Classifying as-is.")
         topic = get_cluster_topic([instances[i] for i in instance_indices])
         resolved_groups[f"{current_key_prefix}_topic_{topic}"] = instance_indices
         return resolved_groups


    # 3. Determine number of splits
    n_splits = math.ceil(len(instance_indices) / config.KMEANS_SPLIT_FACTOR)
    # Ensure n_splits is at least 2 to guarantee a split
    n_splits = max(2, n_splits) 
    print(f"    > Forcing split into {n_splits} sub-clusters.")
    
    # 4. Run K-Means
    try:
        # Use reduced embeddings for K-Means
        kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init='auto')
        sub_cluster_labels = kmeans.fit_predict(reduced_cluster_embeddings)
    except Exception as e:
         print(f"    [ERROR] K-Means failed for splitting '{current_key_prefix}': {e}. Classifying as-is.")
         topic = get_cluster_topic([instances[i] for i in instance_indices])
         resolved_groups[f"{current_key_prefix}_topic_{topic}"] = instance_indices
         return resolved_groups

    
    # 5. Group by new K-Means labels, mapping back to original indices
    sub_groups = group_by_label(sub_cluster_labels, instance_indices_map=instance_indices)

    # --- RECURSION ---
    print(f"    > Split into {len(sub_groups)} sub-clusters. Recursing...")
    for sub_label, sub_instance_indices in sub_groups.items():
        if not sub_instance_indices: continue # Skip empty clusters
            
        new_prefix = f"{current_key_prefix}_sub_{sub_label}"
        
        # Recursive call
        sub_resolved_groups = _recursive_segregate(
            sub_instance_indices, new_prefix, instances, all_embeddings
        )
        resolved_groups.update(sub_resolved_groups)
    
    return resolved_groups

def segregate_clusters_by_topic(instances: List[KeyInfoInstance], initial_groups: Dict[int, List[int]], all_embeddings: np.ndarray,) -> Dict[str, List[int]]:
    """
    (Launcher Function) Splits initial clusters into topical sub-clusters using recursion.
    Handles noise points separately.
    """
    print("\n[Step 3] Segregating initial clusters by topic...")
    all_topical_groups = {}
    
    # Separate noise points (label -1)
    noise_indices = initial_groups.pop(-1, [])
    
    # Process actual clusters found by HDBSCAN
    for label, instance_indices in initial_groups.items():
        key_prefix = f"cluster_{label}"
        resolved_groups = _recursive_segregate(instance_indices, key_prefix, instances, all_embeddings)
        all_topical_groups.update(resolved_groups)

    # Process noise points if they exist
    if noise_indices:
        print("  [Split] Processing noise points as their own group...")
        key_prefix = "cluster_noise"
        resolved_noise_groups = _recursive_segregate(noise_indices, key_prefix, instances, all_embeddings)
        all_topical_groups.update(resolved_noise_groups)
            
    print(f"    > Created {len(all_topical_groups)} topical groups in total after recursion.")
    return all_topical_groups

# --- Temporal Segregation & Validation Logic ---

def _normalize_time_of_day(when_data: Dict[str, Any]) -> str:
    """Extracts a normalized time category (morning, afternoon, etc.)."""
    time_of_day = str(when_data.get("time_of_day", "")).lower().strip()
    time_range = str(when_data.get("time_range", ""))

    # Prefer explicit time_of_day if present and valid
    if time_of_day in ["morning", "afternoon", "evening", "night"]:
        return time_of_day
        
    # Otherwise, try to infer from time_range (HH:MM or HH:MM-HH:MM)
    if time_range:
        # Look for the start time (first HH:MM pattern)
        match = re.search(r"(\d{1,2}):(\d{2})", time_range)
        if match:
            try:
                hour = int(match.group(1))
                # Normalize hour (e.g., handle 24h format if needed, though data seems consistent)
                if 4 <= hour <= 11: return "morning"
                if 12 <= hour <= 17: return "afternoon"
                if 18 <= hour <= 21: return "evening"
                # Night includes late evening and early morning before 4 AM
                if 22 <= hour <= 23 or 0 <= hour <= 3: return "night"
            except (ValueError, IndexError):
                 pass # Ignore if parsing fails

    # Fallback if neither is useful
    return "unknown"


def segregate_clusters_by_time(instances: List[KeyInfoInstance], topical_groups: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Splits topical clusters into candidate groups based on time."""
    print("\n[Step 4] Segregating topical groups by time of day...\nSkipping temporal segregation (not applicable for this dataset)")
    
    """
    final_candidate_groups = defaultdict(list)
    
    for topical_key, instance_indices in topical_groups.items():
        for i in instance_indices:
            instance = instances[i]
            time_group = str(instance.WHEN).strip().replace(" ", "_").lower() # use WHEN field directly for grouping
            group_key = f"{topical_key}_time_{time_group}"
            final_candidate_groups[group_key].append(i) # Store original index
            
    print(f"    > Split {len(topical_groups)} topical groups into {len(final_candidate_groups)} final candidate groups.")
    
    return dict(final_candidate_groups) # Convert back to dict
    """
    return topical_groups


def validate_and_finalize_clusters(instances: List[KeyInfoInstance], candidate_groups: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
    """Applies final validation rules (recurrence, min size)"""
    
    print("\n[Step 5] Validating final candidate groups against consensus rules...")
    final_clusters = {}
    final_cluster_id = 0

    # Sort candidate groups by key for consistent output order
    sorted_group_keys = sorted(candidate_groups.keys())

    for group_key in sorted_group_keys:
        instance_indices = candidate_groups[group_key]
        
        passes_recurrence = len(instance_indices) >= config.FINAL_MIN_RECURRENCE
        passes_size = len(instance_indices) >= config.FINAL_MIN_CLUSTER_SIZE
        
        # --- Apply validation rules ---
        if passes_recurrence and passes_size:
            # Map indices back to original instance IDs (e.g., "keyinfo-X_raw-Y")
            topic_name = "Unknown"
            if "_topic_" in group_key:
                topic_name = group_key.split("_topic_")[1]
                # Cắt bỏ các hậu tố đệ quy nếu có
                topic_name = topic_name.split("_sub_")[0].split("_time_")[0] 

            instance_ids = [instances[i].provenance  for i in instance_indices]
            final_clusters[str(final_cluster_id)] = {
                "topic": topic_name,
                "instances": instance_ids
            }
            
            print(f"  [PASS] Group '{group_key}' PASSED.")
            print(f"    > Created Final Cluster {final_cluster_id} (Size: {len(instance_indices)})")
            final_cluster_id += 1
        else:
            print(f"  [FAIL] Group '{group_key}' FAILED.")
            if not passes_size:
                print(f"    > Reason: Fails minimum size rule (Size: {len(instance_indices)}, Min: {config.FINAL_MIN_CLUSTER_SIZE})")
                
    print(f"  > Created {len(final_clusters)} final, validated clusters.")
    return final_clusters

def save_clusters(clusters: Dict[str, Dict[str, Any]], filepath: str):
    """Saves the final clusters to a JSON file."""
    print(f"\n[Final] Saving {len(clusters)} final clusters to '{filepath}'...")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=2)
        print(f"  > Successfully saved.")
    except Exception as e:
        print(f"  [ERROR] Failed to save output file: {e}")

# --- MAIN PIPELINE FUNCTION ---

def run_clustering_pipeline(instances: List[KeyInfoInstance], user_output_dir: str, dataset_name: str) -> Dict[str, List[str]]:
    """
    Executes the full 5-step consensus clustering pipeline.
    This is the main entry point called by the router.
    """
    if not instances:
        print("[Clustering Pipeline] No instances provided. Aborting.")
        return {}
    model = get_embedding_model()
    if not model:
        print("[Clustering Pipeline] Embedding model not loaded. Aborting.")
        return {}

    # --- STEP 1: Fingerprint & Embed ---
    fingerprints = create_semantic_fingerprints(instances, dataset_name)
    all_embeddings = embed_fingerprints(fingerprints)
    if all_embeddings is None or all_embeddings.size == 0:
        print("[Clustering Pipeline] Failed during Step 1. Aborting.")
        return {}
    print("\n[SUCCESS] Step 1 Complete: Data loaded and embedded.")
    
    # --- STEP 2: Initial Clustering ---
    reduced_embeddings = run_umap(all_embeddings)
    if reduced_embeddings is None:
         print("[Clustering Pipeline] Failed during UMAP in Step 2. Aborting.")
         return {}
         
    initial_cluster_labels = run_hdbscan_clustering(reduced_embeddings, config.HDBSCAN_MIN_CLUSTER_SIZE)
    initial_groups = group_by_label(initial_cluster_labels)
    print("\n[SUCCESS] Step 2 Complete: Initial semantic groups created.")

    # --- STEP 3: Topical Segregation (Recursive) ---
    topical_groups = segregate_clusters_by_topic(
        instances, 
        initial_groups, 
        all_embeddings # Pass original embeddings
    )
    print("\n[SUCCESS] Step 3 Complete: Groups segregated by topic.")

    # --- STEP 4: Temporal Segregation ---
    final_candidate_groups = segregate_clusters_by_time(instances, topical_groups)
    print("\n[SUCCESS] Step 4 Complete: Groups segregated by time.")

    # --- STEP 5: Validate and Finalize ---
    final_clusters = validate_and_finalize_clusters(instances, final_candidate_groups)
    print("\n[SUCCESS] Step 5 Complete: Final clusters validated.")
    
    # --- Save Results ---
    output_file = os.path.join(user_output_dir, 'L1_consensus_clusters.json')
    save_clusters(final_clusters, output_file)
    
    print("\n--- Consensus Clustering Pipeline Finished ---")
    return final_clusters
