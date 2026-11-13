"""
Main entry point for the Cognitive Graph Pipeline (v2).

This script orchestrates the process of building and updating a hierarchical
knowledge graph from raw narrative text.
"""
import os
import json
import random
import re
from typing import List, Dict, Any

import config
import llm_handler
import clustering
from datastructure import KeyInfoInstance, KnowledgeNode, Dimension
import graph_operations as graph_ops
from logging_context import get_current_logger 

def run_phase_1_data_preparation() -> List[KeyInfoInstance]:
    """
    Runs Phase 1, Step 1: Ingests raw text and extracts structured "keyinfo" instances.
    """
    print("\n" + "="*80)
    print("                PHASE 1, STEP 1: DATA INGESTION & PREPARATION")
    print("="*80)

    # Check if the output already exists to avoid re-running
    if os.path.exists(config.KEYINFO_INSTANCES_FILE) and not config.FORCE_REGENERATE_L1:
        print(f"[CACHE] Loading existing keyinfo instances from {config.KEYINFO_INSTANCES_FILE}")
        with open(config.KEYINFO_INSTANCES_FILE, 'r', encoding='utf-8') as f:
            instances_data = json.load(f)
        print(f"[INFO] Loaded {len(instances_data)} instances from cache.")
        return [KeyInfoInstance(**data) for data in instances_data]

    # Read raw data file
    if not os.path.exists(config.RAW_DATA_FILE):
        print(f"[ERROR] Raw data file not found at {config.RAW_DATA_FILE}. Aborting.")
        return []

    with open(config.RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"[INFO] Successfully read raw data from {config.RAW_DATA_FILE}.")

    text_entries = re.split(r'\n\s*---\s*\n', raw_text)

    if len(text_entries) <= 1 and len(raw_text) > 0:
        print(f"[WARN] The data separator '{config.DATA_SEPARATOR}' was not found or did not split the file into multiple entries. The entire file will be processed as a single entry, which may fail.")

    print(f"[INFO] Found {len(text_entries)} raw text entries to process.")

    all_instances: List[KeyInfoInstance] = []
    for i, entry in enumerate(text_entries):
        if not entry.strip():
            continue

        print(f"\n>> Processing Entry {i+1}/{len(text_entries)}...")

        # 1. Extract 5W1H instances
        keyinfo_list = llm_handler.extract_key_info(entry)
        if not keyinfo_list:
            print(f"  [WARN] No key info extracted for entry {i+1}.")
            continue
        print(f"  - Extracted {len(keyinfo_list)} potential instances.")

        # 2. Format time data and assign IDs
        print("  - Formatting time data and assigning final IDs...")
        for j, info in enumerate(keyinfo_list):
            formatted_time = llm_handler.format_time(info.get("WHEN", ""))
            instance = KeyInfoInstance(
                idx=f"keyinfo-{j}_raw-{i}",
                day_id=i,
                type=info.get("type", "Unknown"),
                WHAT=info.get("WHAT", ""),
                WHEN=formatted_time,
                WHERE=info.get("WHERE", ""),
                WHO=info.get("WHO", ""),
                WHY=info.get("WHY", ""),
                HOW=info.get("HOW", "")
            )
            all_instances.append(instance)

    print(f"\n[SUCCESS] Extracted a total of {len(all_instances)} keyinfo instances.")

    # Save the output
    os.makedirs(os.path.dirname(config.KEYINFO_INSTANCES_FILE), exist_ok=True)
    with open(config.KEYINFO_INSTANCES_FILE, 'w', encoding='utf-8') as f:
        json.dump([inst.dict() for inst in all_instances], f, indent=2)

    print(f"[SAVE] Saved keyinfo instances to {config.KEYINFO_INSTANCES_FILE}")
    print("-" * 80)
    print("         PHASE 1, STEP 1: COMPLETED")
    print("-" * 80)
    return all_instances


def run_phase_1_knowledge_graph_creation(instances: List[KeyInfoInstance]) -> List[KnowledgeNode]:
    """
    Runs Phase 1, Steps 2 & 3: Clusters instances and generates the foundational L1 knowledge graph.
    """
    print("\n" + "="*80)
    print("           PHASE 1, STEPS 2 & 3: L1 KNOWLEDGE GRAPH CREATION")
    print("="*80)

    # Check if the output already exists
    if os.path.exists(config.L1_NODES_FILE) and not config.FORCE_REGENERATE_L1:
        print(f"[CACHE] Loading existing L1 Knowledge Nodes from {config.L1_NODES_FILE}")
        nodes = graph_ops.load_graph_layer(1)
        print(f"[INFO] Loaded {len(nodes)} L1 nodes from cache.")
        return nodes

    # Step 2: Thematic Clustering
    print("\n>> Step 2: Running Thematic Clustering...")
    instance_clusters = clustering.run_clustering_pipeline(instances)

    # Step 3: Direct Inference Generation
    print("\n>> Step 3: Generating L1 Inferences directly from clusters...")
    layer1_nodes: List[KnowledgeNode] = []

    cluster_count = len(instance_clusters)
    # FIX: Use enumerate to get a unique index 'i' for each cluster
    for i, cluster_instance_ids in enumerate(instance_clusters.values()):
        print(f"  - Synthesizing Node from Cluster {i+1}/{cluster_count}...")
        # Find the full instance data for the IDs in the cluster
        cluster_instances = [inst for inst in instances if inst.idx in cluster_instance_ids]

        # FIX: Pass the unique index 'i' to the creation function
        node = graph_ops.create_node_from_cluster(cluster_instances, layer=1, node_index=i)
        if node:
            print(f"    - Successfully created Node: '{node.title}' with ID '{node.node_id}'")
            layer1_nodes.append(node)
        else:
            print(f"    [WARN] Failed to generate node for cluster {i+1}.")

    print(f"\n[SUCCESS] Generated {len(layer1_nodes)} Layer 1 Knowledge Nodes.")

    # Save the L1 graph
    graph_ops.save_graph_layer(layer1_nodes, 1)

    print("-" * 80)
    print("         PHASE 1, STEPS 2 & 3: COMPLETED")
    print("-" * 80)
    return layer1_nodes


def run_higher_layer_phase(previous_layer_nodes: List[KnowledgeNode], layer_number: int) -> List[KnowledgeNode]:
    """
    Runs Phase 2: Generates a new, more abstract layer of the knowledge graph.
    Now uses a Cluster-then-Synthesize approach to maximize node generation.
    """
    logger = get_current_logger() # Use the logger context
    
    print("\n" + "="*80)
    print(f"                 PHASE 2: GENERATING LAYER {layer_number}")
    print("="*80)

    # Check if output exists
    output_file = config.GRAPH_LAYER_FILE_TPL.format(layer=layer_number)
    if os.path.exists(output_file) and not config.FORCE_REGENERATE_HIGHER_LAYERS:
        print(f"[CACHE] Loading existing L{layer_number} Knowledge Nodes...")
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [KnowledgeNode(**item) for item in data]

    # Step 4: Dynamic Dimension Generation
    print(f"\n>> Step 4: Generating Dimensions for L{layer_number}...")
    
    sample_size = min(len(previous_layer_nodes), config.DIMENSION_SAMPLE_SIZE)
    sample_nodes = random.sample(previous_layer_nodes, sample_size)
    
    dimensions_dict = llm_handler.generate_dimensions(sample_nodes, layer_number)
    
    layer_key = f"L{layer_number}"
    current_dimensions = dimensions_dict.get(layer_key, [])
    
    print(f"  - Generated {len(current_dimensions)} dimensions for {layer_key}:")
    for dim in current_dimensions:
        print(f"    - '{dim.title}'")

    # Step 5: Dimensional Clustering & Synthesis
    print(f"\n>> Step 5: Clustering and Synthesizing L{layer_number} nodes...")
    new_layer_nodes: List[KnowledgeNode] = []

    for i, dim in enumerate(current_dimensions):
        print(f"\n  - Processing Dimension {i+1}/{len(current_dimensions)}: '{dim.title}'")
        
        # 1. Cluster the previous layer's nodes
        # --- UPDATED: Passing layer_number correctly ---
        clusters = llm_handler.cluster_nodes_by_dimension(previous_layer_nodes, dim, layer=layer_number)
        print(f"    > Found {len(clusters)} clusters for this dimension.")
        
        # 2. Iterate through clusters and synthesize
        for cluster in clusters:
            cluster_label = cluster['label']
            cluster_nodes = cluster['nodes']
            
            if len(cluster_nodes) < 2:
                continue
                
            print(f"      - Synthesizing cluster: '{cluster_label}' ({len(cluster_nodes)} source nodes)...")
            
            # Use the new function to get a LIST of nodes
            nodes_data_list = llm_handler.synthesize_nodes_from_cluster(
                cluster_nodes, 
                cluster_label, 
                dim, 
                layer_number
            )
            
            if nodes_data_list:
                for node_data in nodes_data_list:
                    new_node = KnowledgeNode(
                        node_id="TEMP", # Will assign later
                        layer=layer_number,
                        **node_data
                    )
                    new_layer_nodes.append(new_node)
                    print(f"        > Created Node: {new_node.title}")
            else:
                print(f"        > No insights generated for this cluster.")

    # Final ID Assignment
    print(f"\n[INFO] Assigning final sequential IDs to {len(new_layer_nodes)} new L{layer_number} nodes...")
    for i, node in enumerate(new_layer_nodes):
        node.node_id = f"L{layer_number}_Node_{i}"

    print(f"\n[SUCCESS] Generated a total of {len(new_layer_nodes)} Layer {layer_number} Knowledge Nodes.")

    print("-" * 80)
    print(f"         PHASE 2 (LAYER {layer_number}): COMPLETED")
    print("-" * 80)
    return new_layer_nodes

def update_model_with_new_data():
    """
    Runs Phase 3: Ingests new data and updates the existing L1 knowledge graph.
    """
    print("\n" + "="*80)
    print("                PHASE 3: MODEL ALIGNMENT & UPDATE")
    print("="*80)
    if not os.path.exists(config.NEW_DATA_FILE):
        print(f"[INFO] No new data file found at '{config.NEW_DATA_FILE}'. Skipping update phase.")
        return

    # Load existing L1 graph
    if not os.path.exists(config.L1_NODES_FILE):
        print("[ERROR] L1 graph not found. Please run the initial generation first before updating.")
        return

    print(f"[INFO] Loading existing L1 graph from '{config.L1_NODES_FILE}'...")
    existing_l1_nodes = graph_ops.load_graph_layer(1)

    # Step 6: Ingest and Cluster New Data
    print("\n>> Step 6: Ingesting and clustering new data entries...")
    with open(config.NEW_DATA_FILE, 'r', encoding='utf-8') as f:
        new_text = f.read()

    print("  - Extracting keyinfo from new entry...")
    new_instances_raw = llm_handler.extract_key_info(new_text)
    new_instances = [KeyInfoInstance(idx=f"new-keyinfo-{i}", day_id=999, **info) for i, info in enumerate(new_instances_raw)]
    print(f"  - Extracted {len(new_instances)} new instances.")

    # Step 7: Knowledge Assimilation
    print("\n>> Step 7: Assimilating new knowledge by finding similarity...")
    most_similar_node, similarity = graph_ops.find_most_similar_node(new_instances, existing_l1_nodes)

    # Step 8: Refinement or Creation
    print("\n>> Step 8: Refining existing knowledge or creating new node...")
    if most_similar_node and similarity > config.SIMILARITY_THRESHOLD:
        print(f"  - Found a similar node: '{most_similar_node.title}' (Similarity: {similarity:.2f}).")
        print("  - Refining this node with the new information...")
        updated_node = llm_handler.refine_node(most_similar_node, new_instances)

        # Update the node in our list
        for i, node in enumerate(existing_l1_nodes):
            if node.node_id == updated_node.node_id:
                existing_l1_nodes[i] = updated_node
                break
        print("  - [SUCCESS] Node content has been updated.")

    else:
        if most_similar_node:
            print(f"  - Most similar node ('{most_similar_node.title}') similarity ({similarity:.2f}) is below threshold ({config.SIMILARITY_THRESHOLD}).")
        print("  - Creating a new L1 node for this novel pattern...")
        # FIX: Pass the new unique index for the node ID
        new_node_index = len(existing_l1_nodes)
        new_node = graph_ops.create_node_from_cluster(new_instances, layer=1, node_index=new_node_index)
        if new_node:
            existing_l1_nodes.append(new_node)
            print(f"  - [SUCCESS] Created new node: '{new_node.title}'.")

    # Save the updated L1 graph
    graph_ops.save_graph_layer(existing_l1_nodes, 1)
    print("-" * 80)
    print("         PHASE 3: COMPLETED")
    print("-" * 80)

if __name__ == "__main__":
    print("*"*80)
    print("* STARTING COGNITIVE GRAPH PIPELINE                *")
    print("*"*80)

    # --- Initial Knowledge Graph Generation ---
    # Phase 1: Create L1 from raw data
    keyinfo_instances = run_phase_1_data_preparation()
    if keyinfo_instances:
        l1_nodes = run_phase_1_knowledge_graph_creation(keyinfo_instances)

        # Phase 2: Create higher layers iteratively
        current_layer_nodes = l1_nodes
        for i in range(2, config.TOTAL_LAYERS + 1):
            if not current_layer_nodes:
                print(f"[ERROR] Cannot generate Layer {i} because Layer {i-1} is empty. Stopping.")
                break
            current_layer_nodes = run_higher_layer_phase(current_layer_nodes, layer_number=i)

        print("\n\n--- PIPELINE COMPLETED ---")
    else:
        print("\n\n--- PIPELINE HALTED: No keyinfo instances were generated. ---")


    # --- Optional: Update the Model with New Data ---
    if config.RUN_UPDATE_PROCESS:
        update_model_with_new_data()

    print("\n" + "*"*80)
    print("* PIPELINE EXECUTION FINISHED                     *")
    print("*"*80)
