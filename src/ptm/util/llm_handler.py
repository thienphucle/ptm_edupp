import os
import json
import time
import re
import random
import importlib
from pydantic import ValidationError
from typing import List, Dict, Any, Optional
import traceback # For logging exceptions
from .datastructure import KeyInfoInstance, KnowledgeNode, Dimension
from . import prompts, config


# --- NEW IMPORTS ---
# Import the client library functions
from .api_client import query_api_with_retries, MODEL_LITE, MODEL_PRO
from .logging_context import get_current_logger


def extract_key_info(record: Dict[str, Any], prompt_template: str) -> List[KeyInfoInstance]:
    """
    Extracts structured 5W1H "keyinfo" from a structured data record.
    
    Args:
        record: Dictionary chứa dữ liệu gốc (VD: user_id, source_code, time_taken...)
        prompt_template: Chuỗi prompt tương ứng
        
    Returns:
        List[KeyInfoInstance]: Danh sách các object Pydantic đã được validate.
    """
    logger = get_current_logger()
    task = "Extract Key Information"

    try:
        prompt = prompt_template.format(**record)
    except KeyError as e:
        logger.error(f"  [Prompt Error] Missing key in record for prompt formatting: {e}")
        return []

    try:
        time.sleep(5) 
        data = query_api_with_retries(
            prompt=prompt, 
            model=MODEL_LITE, 
            outjson=True
        )

        raw_infos = []
        if isinstance(data, list):
            raw_infos = data
        elif isinstance(data, dict) and "informations" in data and isinstance(data["informations"], list):
            raw_infos = data["informations"]
        elif isinstance(data, dict) and "WHAT" in data:
            logger.warning("  [WARN] LLM returned a single JSON object. Wrapping it.")
            raw_infos = [data]
        else:
            logger.warning(f"  [WARN] Could not extract key info. Parsed format not recognized.")
            return []

        validated_instances = []
        for idx, info in enumerate(raw_infos):
            try:
                # Đảm bảo provenance không bị thiếu (Fallback nếu LLM quên)
                if "provenance" not in info or not info["provenance"]:
                    # Gán fallback dựa trên ID của record (tùy thuộc dataset)
                    if "submission_id" in record:
                        info["provenance"] = str(record["submission_id"])
                    elif "user_id" in record and "question_id" in record:
                        info["provenance"] = f"U{record['user_id']}_Q{record['question_id']}"
                
                # Gán user_id tự động từ record để tránh LLM bịa
                info["user_id"] = str(record.get("user_id", "unknown"))

                # Khởi tạo Pydantic Model (sẽ tự động bắt lỗi nếu sai kiểu dữ liệu)
                node = KeyInfoInstance(**info)
                validated_instances.append(node)

            except ValidationError as ve:
                logger.warning(f"  [Validation Error] L0 Node {idx} failed validation: {ve.errors()}")
            except Exception as ex:
                logger.warning(f"  [Parse Error] Could not parse L0 Node {idx}: {ex}")

        return validated_instances

    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task}': {e}", exc_info=True)
        return []

def format_time(time_string: str) -> Dict[str, Any]:
    """
    Formats a single time string.
    (Refactored to use query_api_with_retries)
    """
    logger = get_current_logger()
    if not time_string: return {}
    task = "Format Time String"
    prompt = prompts.ACCURATE_TIME_FORMATTING.format(time_string=time_string)
    
    try:
        time.sleep(4)
        data = query_api_with_retries(
            prompt=prompt, 
            model=MODEL_LITE, 
            outjson=True
        )
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task}': {e}", exc_info=True)
        return {}


def format_time_batch(time_strings: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Formats a batch of time strings using a single LLM call.
    (Refactored to use query_api_with_retries)
    """
    logger = get_current_logger()
    if not time_strings: return {}

    task = "Format Time Strings Batch"
    time_strings_json = json.dumps(time_strings, indent=2)
    prompt = prompts.ACCURATE_TIME_FORMATTING_BATCH.format(time_strings_json=time_strings_json)
    
    try:
        time.sleep(4)
        data = query_api_with_retries(
            prompt=prompt,
            model=MODEL_LITE,
            outjson=True
        )
        
        if isinstance(data, dict):
            result = {key: data.get(key, {}) for key in time_strings}
            return result
            
        logger.error(f"  [LLM Call] Batch time formatting failed to return a valid dict. Returning empty results.")
        return {key: {} for key in time_strings}
    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task}': {e}", exc_info=True)
        return {key: {} for key in time_strings}


def generate_dimensions(sampled_nodes: List[KnowledgeNode], layer_number: int) -> Dict[str, List[Dimension]]:
    """
    Generates analytical dimensions for all higher layers (L2, L3, L4) at once.
    """
    logger = get_current_logger()
    task = f"Generate All Hierarchical Dimensions from L{layer_number}"
    sampled_nodes_text = "\n".join([f"- {node.title}: {node.content}" for node in sampled_nodes])
    num_dimensions_per_layer = config.NUM_DIMENSIONS_PER_LAYER 
    
    prompt = prompts.DIMENSION_GENERATION.format(
        layer_number=layer_number,
        num_dimensions=num_dimensions_per_layer,
        sampled_nodes_text=sampled_nodes_text
    )
    
    all_dimensions: Dict[str, List[Dimension]] = {"L2": [], "L3": [], "L4": []}

    try:
        time.sleep(4)
        data = query_api_with_retries(prompt=prompt, model=MODEL_PRO, outjson=True)
        
        
        if isinstance(data, dict) and "L2" in data and "L3" in data and "L4" in data:
            logger.info(f"  [LLM Parse] Successfully parsed hierarchical dimension object.")
            for layer_key in ["L2", "L3", "L4"]:
                layer_dims_data = data.get(layer_key, [])
                if isinstance(layer_dims_data, list):
                    for dim_data in layer_dims_data:
                        if isinstance(dim_data, dict) and "title" in dim_data and "description" in dim_data:
                            all_dimensions[layer_key].append(Dimension(**dim_data))
                        else:
                            logger.warning(f"  [WARN] Malformed dimension data in {layer_key} skipped: {dim_data}")
                else:
                        logger.warning(f"  [WARN] Data for {layer_key} was not a list. Skipping.")
            logger.info(f"  - Generated {len(all_dimensions['L2'])} dimensions for L2.")
            return all_dimensions
        
        return all_dimensions 

    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task}': {e}", exc_info=True)
        return all_dimensions 


def refine_node(existing_node: KnowledgeNode, new_instances: List[KeyInfoInstance]) -> KnowledgeNode:
    """
    Updates an existing KnowledgeNode with new information.
    """
    logger = get_current_logger()
    task = f"Refine Node '{existing_node.node_id}'"
    new_instances_text = "\n".join([f"- {inst.WHAT}" for inst in new_instances])
    prompt = prompts.NODE_REFINEMENT.format(
        existing_node_content=existing_node.content,
        new_instances_text=new_instances_text
    )
    
    try:
        time.sleep(4)
        data = query_api_with_retries(prompt=prompt, model=MODEL_LITE, outjson=True)
        
        if data and "updated_content" in data:
            logger.info(f"    - Successfully received refined content for node.")
            existing_node.content = data["updated_content"]            
            new_source_ids = [inst.provenance for inst in new_instances]
            for sid in new_source_ids:
                if sid not in existing_node.source_instances:
                    existing_node.source_instances.append(sid)
        else:
            logger.warning(f"    [WARN] Failed to get refined content. Node remains unchanged.")
                    
    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task}': {e}", exc_info=True)
        logger.warning(f"    [WARN] Failed to get refined content due to error. Node remains unchanged.")

    return existing_node

def get_cluster_topic(cluster_instances: List[KeyInfoInstance]) -> str:
    """
    Uses an LLM to assign a single topic category to a cluster.
    (Refactored to use query_api_with_retries)
    """
    logger = get_current_logger()
    
    instances_text = "\n".join([f"- {inst.WHAT}" for inst in cluster_instances[:20]])
    prompt = prompts.TOPIC_CLASSIFICATION_PROMPT.format(instances_text=instances_text)
    task_description = "Classify Cluster Topic"

    try:
        time.sleep(4)
        response_text = query_api_with_retries(
            prompt=prompt,
            model=MODEL_PRO,
            outjson=False
        )

        if response_text:
            topic = re.sub(r"^\s*Topic:\s*", "", response_text, flags=re.IGNORECASE).strip()
            topic = re.sub(r"\s+", "_", topic) 
            return topic if topic else "Other"
            
    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task_description}': {e}", exc_info=True)
        
    logger.error(f"    [LLM Topic] Failed to classify topic. Defaulting to 'Other'.")
    return "Other"



# --- UPDATED: Cluster nodes based on a Dimension with Configurable 'm' ---
def cluster_nodes_by_dimension(all_nodes: List[KnowledgeNode], dimension: Dimension, layer: int) -> List[Dict[str, Any]]:
    """
    Uses LLM to group nodes into clusters based on a specific dimension.
    Returns a list of clusters, where each cluster has a 'label' and a list of 'nodes'.
    """
    logger = get_current_logger()
    task_desc = f"Cluster nodes by dimension '{dimension.title}'"
    
    if not all_nodes:
        return []

    nodes_to_process = all_nodes

    # 1. Create numbered list mapping
    node_map = {}
    prompt_list = []
    for i, node in enumerate(nodes_to_process):
        index = i + 1 
        prompt_list.append(f"{index}. [ID: {node.node_id}] {node.title}: {node.content}")
        node_map[index] = node
    
    numbered_nodes_text = "\n".join(prompt_list)

    # 2. Determine 'm' (number of clusters)
    # --- START NEW LOGIC ---
    target_m = 0
    
    # A. Check Per-Layer Config first
    layer_targets = getattr(config, 'TARGET_CLUSTERS_BY_LAYER', {})
    layer_key = f"L{layer}"
    if layer_targets and layer_key in layer_targets and layer_targets[layer_key] > 0:
        target_m = layer_targets[layer_key]
        logger.info(f"    [Config] Using specific target m={target_m} for {layer_key}.")
    
    # B. Fallback to Global Config
    if target_m == 0:
        target_m = getattr(config, 'TARGET_CLUSTERS_PER_DIMENSION', 0)
    
    if target_m > 0:
        # Use fixed user setting (clamped)
        num_clusters = min(target_m, len(nodes_to_process))
        num_clusters = max(2, num_clusters) if len(nodes_to_process) >= 2 else 1
        if target_m != num_clusters:
             logger.info(f"    [Config] Target m={target_m} clamped to {num_clusters} due to available nodes.")
    else:
        # Use Auto Heuristic: ~1 cluster for every 4 nodes, minimum 2, maximum 8
        num_clusters = max(2, min(8, len(nodes_to_process) // 4))
        logger.info(f"    [Auto] Calculated m={num_clusters} clusters based on {len(nodes_to_process)} nodes.")
    # --- END NEW LOGIC ---

    prompt = prompts.DIMENSIONAL_CLUSTERING_PROMPT.format(
        dimension_title=dimension.title,
        dimension_description=dimension.description,
        num_clusters=num_clusters, 
        numbered_nodes_text=numbered_nodes_text
    )

    clusters_output = []

    try:
        time.sleep(4)
        data = query_api_with_retries(
            prompt=prompt,
            model=MODEL_PRO,
            outjson=True
        )

        if isinstance(data, dict) and "clusters" in data and isinstance(data["clusters"], list):
            raw_clusters = data["clusters"]
            
            for rc in raw_clusters:
                label = rc.get("cluster_label", "Unnamed Group")
                indices = rc.get("node_indices", [])
                
                cluster_nodes = []
                for idx in indices:
                    if idx in node_map:
                        cluster_nodes.append(node_map[idx])
                
                if len(cluster_nodes) >= 2: 
                    clusters_output.append({
                        "label": label,
                        "nodes": cluster_nodes
                    })
            
            logger.info(f"  - Dimension '{dimension.title}' generated {len(clusters_output)} clusters.")
            return clusters_output
        else:
            logger.warning(f"  [WARN] Clustering for '{dimension.title}' returned invalid format.")
            return []

    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task_desc}': {e}", exc_info=True)
        return []



# --- UPDATED FUNCTION: Synthesize MULTIPLE nodes from a specific Cluster ---
def synthesize_nodes_from_cluster(cluster_nodes: List[KnowledgeNode], cluster_label: str, dimension: Dimension, layer: int) -> List[Dict[str, Any]]:
    """
    Generates ONE OR MORE higher-layer nodes from a specific cluster of source nodes.
    Returns a list of node data dictionaries.
    """
    logger = get_current_logger()
    
    if not cluster_nodes:
        return []

    source_nodes_json = json.dumps(
        [{'node_id': n.node_id, 'title': n.title, 'content': n.content} for n in cluster_nodes], 
        indent=2
    )
    
    prompt = prompts.GRAPH_BASED_SYNTHESIS.format(
        dimension_title=dimension.title,
        dimension_description=dimension.description,
        cluster_label=cluster_label,
        source_nodes_json=source_nodes_json
    )
    
    task_desc = f"Synthesize L{layer} nodes from cluster '{cluster_label}'"
    generated_nodes = []

    try:
        time.sleep(4)
        data = query_api_with_retries(
            prompt=prompt,
            model=MODEL_PRO, 
            outjson=True
        )

        parsed_items = []
        if isinstance(data, list):
            parsed_items = data
        # Case 1: Expected List of objects
        elif isinstance(data, dict):
            for key in ["patterns", "nodes", "insights", "results"]:
                if key in data and isinstance(data[key], list):
                    parsed_items = data[key]
                    break
                
            if not parsed_items and all(k in data for k in ["title", "content"]):
                parsed_items = [data]
            
        valid_ids = [n.node_id for n in cluster_nodes]
        for item in parsed_items:
            if isinstance(item, dict) and "title" in item and "content" in item:
                returned_ids = item.get("source_nodes", [])
                final_source_ids = [nid for nid in returned_ids if nid in valid_ids]
                if not final_source_ids:
                    final_source_ids = valid_ids
                
                generated_nodes.append({
                    "title": item["title"],
                    "content": item["content"],
                    "source_nodes": final_source_ids,
                    "generating_dimension": dimension.title
                })
                
        return generated_nodes
        

    except Exception as e:
        logger.error(f"  [LLM Call] Error processing task '{task_desc}': {e}", exc_info=True)

    return generated_nodes


#NEW UPDATE FOR CODESTREAM PROBLEMS (extract problems concept before extracting 5W1H)

def extract_all_problem_concepts(problems_df, cache_file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Extracts the "Problem Concept" for each unique problem in the dataset using an LLM.
    Caches the results in a JSON file to avoid repeated API calls.
    """
    logger = get_current_logger()
    
    # 1. Check if cache exists
    if os.path.exists(cache_file_path):
        logger.info(f"  [Cache] Loading problem concepts from {cache_file_path}")
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    logger.info("  [Concept Extraction] Running LLM to extract concepts from problem descriptions...")
    
    concepts_map: Dict[str, Dict[str, str]] = {}

    default_fallback = {
        "paradigm": "Unknown",
        "specific_technique": "Unknown",
        "core_challenge": "No description provided."
    }
    
    # 2. Iterate through each unique problem
    # Assume problems_df is a DataFrame with columns 'problem_id' and 'problem_description'
    for _, row in problems_df.iterrows():
        prob_id = str(row['problem_id'])
        desc = str(row.get('problem_description', ''))
        
        if prob_id in concepts_map:
            continue
            
        if not desc.strip():
            concepts_map[prob_id] = default_fallback
            continue
            
        prompt = prompts.PROBLEM_CONCEPT_EXTRACTION.format(problem_description=desc)
        
        try:
            time.sleep(4) 
            data = query_api_with_retries(
                prompt=prompt,
                model=MODEL_PRO, 
                outjson=True
            )
            
            # check if the returned data has the expected keys
            if data and isinstance(data, dict) and all(k in data for k in ["paradigm", "specific_technique", "core_challenge"]):
                concepts_map[prob_id] = {
                    "paradigm": data["paradigm"],
                    "specific_technique": data["specific_technique"],
                    "core_challenge": data["core_challenge"]
                }
                logger.info(f"    - {prob_id}: [{data['paradigm']}] {data['specific_technique']}")
            else:
                raise ValueError("Invalid schema from LLM")
                
        except Exception as e:
            logger.warning(f"    - {prob_id}: LLM failed ({e}). Using text snippet fallback.")
            
            concepts_map[prob_id] = {
                "paradigm": "Uncategorized",
                "specific_technique": "Original Problem Text",
                "core_challenge": desc.replace('\n', ' ').strip()
            }
            
    # 3. save to cache file
    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
    with open(cache_file_path, 'w', encoding='utf-8') as f:
        json.dump(concepts_map, f, indent=2, ensure_ascii=False)
        
    logger.info(f"  [Concept Extraction] Saved {len(concepts_map)} problem concepts to {cache_file_path}")
    return concepts_map

