import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy.spatial.distance import cosine

from .api_client import query_api_with_retries, MODEL_LITE
from .logging_context import get_current_logger

from . import config, llm_handler, prompts
from .datastructure import KnowledgeNode, KeyInfoInstance, Dimension

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            print("[Graph Ops] Loading sentence transformer model (lazy)...")
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            print("[Graph Ops] Model loaded.")
        except Exception as e:
            print(f"[Graph Ops] WARNING: Could not load embedding model. Similarity search disabled. Error: {e}")
    return embedding_model

def create_nodes_from_cluster(instances: List[KeyInfoInstance], layer: int, cluster_index: int, user_id: str, cluster_topic: str = "General") -> List[KnowledgeNode]:
    """
    Generates one or more KnowledgeNodes from a cluster of instances using an LLM call.
    The prompt now supports returning 1-3 nodes per cluster.
    (Refactored to use query_api_with_retries)
    """
    logger = get_current_logger() # Get the logger

    # --- START NEW FIX ---
    # Filter instances to only include those with actual content.
    # The 'WHAT' field is the most critical for this prompt.
    valid_instances = [inst for inst in instances if inst.WHAT and inst.WHAT.strip()]
    
    if not valid_instances:
        logger.warning(f"  [WARN] Skipping cluster {cluster_index} because it contains no instances with valid 'WHAT' content.")
        return []
    # --- END NEW FIX ---

    # --- Use valid_instances from now on ---
    instance_ids = [inst.provenance for inst in valid_instances]

    # --- Create comprehensive instance text with all 5W1H info ---
    instances_text_parts = []
    for inst in valid_instances: # <-- Use valid_instances
        full_info_parts = []
        if inst.WHAT: full_info_parts.append(f"WHAT: {inst.WHAT}")
        if inst.WHEN: full_info_parts.append(f"WHEN: {inst.WHEN}")
        if inst.WHERE: full_info_parts.append(f"WHERE: {inst.WHERE}")
        if inst.WHO: full_info_parts.append(f"WHO: {inst.WHO}")
        if inst.WHY: full_info_parts.append(f"WHY: {inst.WHY}")
        if inst.HOW: full_info_parts.append(f"HOW: {inst.HOW}")

        full_info = ", ".join(full_info_parts)
        instances_text_parts.append(f"- (ID: {inst.provenance}) {full_info}")
    
    instances_text = "\n".join(instances_text_parts)
    # --- END ---

    prompt = prompts.DIRECT_INFERENCE_GENERATION.format(
        instance_ids_json=json.dumps(instance_ids),
        instances_text=instances_text,
        cluster_topic=cluster_topic
    )
    task_desc = f"Synthesize L{layer} Node(s) from cluster {cluster_index}"

    # --- START REFACTOR ---
    # Replace _call_llm and _parse_json_response with a single call
    
    new_nodes: List[KnowledgeNode] = []
    
    try:
        time.sleep(1)
        data = query_api_with_retries(
            prompt=prompt,
            model=MODEL_LITE,
            outjson=True
        )
        
        parsed_items = []
        
        if isinstance(data, dict):
            # 1. Check for known keys that contain lists of nodes
            for key in ["patterns", "nodes", "informations", "results"]:
                if key in data and isinstance(data[key], list):
                    parsed_items = data[key]
                    break
            # 2. Fallback: If the LLM returns a single object with "title" and "content", treat it as a single node
            if not parsed_items and all(k in data for k in ["title", "content"]):
                parsed_items = [data]
                
        elif isinstance(data, list):
            # 3. If the LLM directly returns an array (not wrapped in an object)
            parsed_items = data
            
        # 4. Filter out any non-dictionary items from the parsed list
        parsed_items = [item for item in parsed_items if isinstance(item, dict)]

        if not parsed_items:
            logger.warning(f"  [WARN] No valid dictionary items found in LLM response for cluster {cluster_index}. Data: {data}")
            return []

        for item in parsed_items:
            if all(k in item for k in ["title", "content", "source_instances"]):
                new_nodes.append(KnowledgeNode(
                    node_id=f"L{layer}_Node_TEMP", 
                    user_id=user_id, 
                    layer=layer,
                    title=item["title"],
                    content=item["content"],
                    source_instances=item["source_instances"],
                    source_nodes=[]
                ))
            else:
                logger.warning(f"  [WARN] Malformed node skipped: {item}")
                    
    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task_desc}': {e}", exc_info=True)
        return []

    return new_nodes


    # --- END REFACTOR ---


# --- REMOVED FUNCTION ---
# This function is no longer called by the router (app/routers/models.py).
# The router now correctly calls `llm_handler.synthesize_nodes_for_dimension` instead.
# This function is now dead code and can be removed.
#
# def create_nodes_from_synthesis(source_nodes: List[KnowledgeNode], dimension: Dimension, layer: int) -> List[KnowledgeNode]:
#     ...
# --- END REMOVED FUNCTION ---


def save_graph_layer(nodes: List[KnowledgeNode], layer: int, output_dir: str):
    filepath = os.path.join(output_dir, f"L{layer}_nodes.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([node.model_dump() for node in nodes], f, indent=2)
    print(f"[SAVE] Saved L{layer} graph with {len(nodes)} nodes to '{filepath}'")

def save_dimensions_for_layer(dimensions: List[Dimension], layer: int, output_dir: str):
    filepath = os.path.join(output_dir, f"L{layer}_dimensions.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([dim.model_dump() for dim in dimensions], f, indent=2)
    print(f"[SAVE] Saved {len(dimensions)} dimensions for L{layer} to '{filepath}'")

def load_graph_layer(layer: int, user_output_dir: str) -> List[KnowledgeNode]:
    """Loads a list of KnowledgeNodes from a JSON file for a specific layer."""
    # --- FIX: Use user_output_dir, not global config ---
    filepath = os.path.join(user_output_dir, f"L{layer}_nodes.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [KnowledgeNode(**item) for item in data]

def find_most_similar_node(new_instances: List[KeyInfoInstance], existing_nodes: List[KnowledgeNode]) -> Tuple[Optional[KnowledgeNode], float]:
    """
    Finds the most semantically similar existing node to a new cluster of instances.
    (Phase 3, Step 7)
    """
    logger = get_current_logger() # Get logger
    if not new_instances or not existing_nodes:
        return None, 0.0
        
    model = get_embedding_model()
    if not model:
        logger.error("[Graph Ops] Similarity search failed: Embedding model not loaded.")
        return None, 0.0

    logger.info(f"  - Comparing {len(new_instances)} new instance(s) against {len(existing_nodes)} existing L1 nodes.")
    # Create a summary of the new instances to embed
    new_cluster_summary = ". ".join([inst.WHAT for inst in new_instances])

    # Get embeddings
    new_embedding = model.encode(new_cluster_summary)
    existing_embeddings = model.encode([node.content for node in existing_nodes])

    # Calculate cosine similarities (1 - cosine distance)
    similarities = 1 - np.array([cosine(new_embedding, emb) for emb in existing_embeddings])

    best_match_index = np.argmax(similarities)
    max_similarity = similarities[best_match_index]

    logger.info(f"    - Best match is node '{existing_nodes[best_match_index].title}' with similarity score: {max_similarity:.4f}")

    return existing_nodes[best_match_index], float(max_similarity)