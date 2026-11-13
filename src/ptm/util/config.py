# This config is specific to the model building pipeline.
# It's kept separate from the main app's config.

import os

# --- Path Configuration ---
OUTPUT_DIR = "app/static/datasets"


# --- Pipeline Control ---
TOTAL_LAYERS = 4 


# --- LLM & API Configuration ---
MODEL_LITE = "gpt-4o-mini"  # For Extraction, Clustering (fast and cost-effective)
MODEL_PRO = "gpt-4o"  # For Synthesis, Dimension Generation (more capable)
MAX_OUTPUT_TOKENS = 8192  
TEMPERATURE = 0.0 

# --- Clustering Configuration ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- Initial Clustering (Step 2) ---
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 5
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER_SIZE = 3 

# --- Recursive Re-clustering (Step 3) ---
MAX_CLUSTER_SIZE = 12
KMEANS_SPLIT_FACTOR = 6

# --- Validation Rules (Step 5) ---
FINAL_MIN_RECURRENCE = 2 
FINAL_MIN_CLUSTER_SIZE = 2 

# --- Feature Weights (weighted by importance) ---
# 1 = normal, 2 = important, 3 = critical
FEATURE_WEIGHTS = {
    "codestream": {"WHAT": 3, "HOW": 3, "WHERE": 1},
    "upsc2m": {"WHAT": 3, "HOW": 3, "WHERE": 2},
    "foundational_assist": {"WHAT": 2, "HOW": 2, "WHERE": 2}
}

# --- Higher-Layer Synthesis Configuration ---
DIMENSION_SAMPLE_SIZE = 30 
NUM_DIMENSIONS_PER_LAYER = 3 

# --- NEW: Cluster Count Control ---
# 0 = Auto (Heuristic)
TARGET_CLUSTERS_PER_DIMENSION = 0
# Per-Layer Overrides
TARGET_CLUSTERS_BY_LAYER = {
    "L2": 4,
    "L3": 3,
    "L4": 2
}

# --- Model Update/Alignment Configuration ---
SIMILARITY_THRESHOLD = 0.67