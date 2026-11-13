import os
import json
import random
from typing import List, Dict, Any
import logging

# Import the necessary modules from the ptm package
from . import config, llm_handler, clustering, graph_operations as graph_ops, prompts
from .logging_context import user_logger_context, get_current_logger
from .dataloader import run_dataloader_test
from .datastructure import KeyInfoInstance, KnowledgeNode, Dimension

def create_user_logger(dataset_name: str, user_id: str, base_log_dir: str = "logs") -> Any:
    import logging
    user_log_dir = os.path.join(base_log_dir, dataset_name, str(user_id))
    os.makedirs(user_log_dir, exist_ok=True)
    
    logger_name = f"ptm_{dataset_name}_{user_id}"
    user_logger = logging.getLogger(logger_name)
    user_logger.setLevel(logging.DEBUG)
    
    if not user_logger.handlers:
        file_handler = logging.FileHandler(os.path.join(user_log_dir, "pipeline.log"), encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        user_logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(f'[{dataset_name.upper()}:{user_id}] %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        user_logger.addHandler(console_handler)
        
    return user_logger, user_log_dir

def process_single_user(dataset_name: str, user_id: str, records: list, user_output_dir: str):
    """
    Chạy toàn bộ Pipeline (L0 -> L1 -> L2 -> L3 -> L4) cho MỘT user duy nhất.
    """
    logger = get_current_logger()
    logger.info(f"--- Starting processing for User {user_id} ({dataset_name}) ---")
    
    # ==========================================
    # PHASE 1: L0 EXTRACTION
    # ==========================================
    logger.info(">> Phase 1: Extracting L0 Nodes...")
    l0_nodes: List[KeyInfoInstance] = []
    
    if dataset_name == "codestream":
        prompt_template = prompts.KEY_INFO_EXTRACTION_CODESTREAM
    elif dataset_name == "upsc2m":
        prompt_template = prompts.KEY_INFO_EXTRACTION_UPSC2M
    else:
        prompt_template = prompts.KEY_INFO_EXTRACTION_UPSC2M 
        
    for record in records:
        # Chuyển Pydantic Record thành Dict
        record_dict = record.model_dump() if hasattr(record, 'model_dump') else record
        
        # Gọi hàm trích xuất 
        extracted_nodes = llm_handler.extract_key_info(record_dict, prompt_template)
        l0_nodes.extend(extracted_nodes)
        
    logger.info(f"   -> Extracted {len(l0_nodes)} L0 nodes.")
    
    # Lưu L0 ra file
    l0_path = os.path.join(user_output_dir, "L0_nodes.json")
    with open(l0_path, 'w', encoding='utf-8') as f:
        json.dump([n.model_dump() for n in l0_nodes], f, indent=2, ensure_ascii=False)
        
    if not l0_nodes:
        logger.warning(" No L0 nodes were extracted. Skipping further processing for this user.")
        return

    # ==========================================
    # PHASE 2: L1 SYNTHESIS (Clustering)
    # ==========================================
    logger.info(">> Phase 2: Clustering & Synthesizing L1 Nodes...")
    
    # Chạy thuật toán Clustering (HDBSCAN + Consensus)
    instance_clusters = clustering.run_clustering_pipeline(l0_nodes, user_output_dir, dataset_name)
    logger.info(f"   -> Generated {len(instance_clusters)} clusters for L0.")
    
    l1_nodes: List[KnowledgeNode] = []
    for i, cluster_data in enumerate(instance_clusters.values()):
        cluster_ids = cluster_data["instances"]
        cluster_topic = cluster_data["topic"]
        cluster_instances = [inst for inst in l0_nodes if inst.provenance in cluster_ids]
        
        # Gọi hàm tạo L1 Node (Đã Refactor để nhận user_id)
        nodes = graph_ops.create_nodes_from_cluster(
            instances=cluster_instances, 
            layer=1, 
            cluster_index=i, 
            user_id=user_id,
            cluster_topic=cluster_topic
        )
        l1_nodes.extend(nodes)
        
    # Gán ID chính thức cho L1
    for i, node in enumerate(l1_nodes):
        node.node_id = f"L1_{user_id}_{i}"
        
    logger.info(f"   -> Synthesized {len(l1_nodes)} L1 nodes.")
    
    l1_path = os.path.join(user_output_dir, "L1_nodes.json")
    with open(l1_path, 'w', encoding='utf-8') as f:
        json.dump([n.model_dump() for n in l1_nodes], f, indent=2, ensure_ascii=False)

    # ==========================================
    # PHASE 3: HIGHER LAYERS (L2 -> L4)
    # ==========================================
    current_layer_nodes = l1_nodes
    
    # Lặp từ Layer 2 đến Layer 4
    for layer_num in range(2, 5):
        logger.info(f">> Phase 3: Synthesizing Layer {layer_num}...")
        
        if not current_layer_nodes:
            logger.warning(f"   Layer {layer_num-1} is empty. Cannot create Layer {layer_num}. Stopping.")
            break
            
        # 1. Sinh Dimensions
        sample_size = min(len(current_layer_nodes), config.DIMENSION_SAMPLE_SIZE)
        sample_nodes = random.sample(current_layer_nodes, sample_size)
        dimensions_dict = llm_handler.generate_dimensions(sample_nodes, layer_number=layer_num-1)
        
        layer_key = f"L{layer_num}"
        current_dimensions = dimensions_dict.get(layer_key, [])
        logger.info(f"   -> Generated {len(current_dimensions)} Dimensions for {layer_key}.")
        
        new_layer_nodes = []
        
        # 2. Cluster & Synthesize theo từng Dimension
        for dim in current_dimensions:
            clusters = llm_handler.cluster_nodes_by_dimension(current_layer_nodes, dim, layer=layer_num)
            
            for cluster in clusters:
                if len(cluster['nodes']) < 2:
                    continue # Bỏ qua cụm 1 node (theo chuẩn PTM)
                    
                nodes_data_list = llm_handler.synthesize_nodes_from_cluster(
                    cluster['nodes'], cluster['label'], dim, layer_num
                )
                
                for node_data in nodes_data_list:
                    new_node = KnowledgeNode(
                        node_id="TEMP",
                        user_id=user_id, # Bắt buộc phải có
                        layer=layer_num,
                        **node_data
                    )
                    new_layer_nodes.append(new_node)
                    
        # Gán ID chính thức
        for i, node in enumerate(new_layer_nodes):
            node.node_id = f"L{layer_num}_{user_id}_{i}"
            
        logger.info(f"   -> Synthesized {len(new_layer_nodes)} L{layer_num} nodes.")
        
        # Lưu file
        layer_path = os.path.join(user_output_dir, f"L{layer_num}_nodes.json")
        with open(layer_path, 'w', encoding='utf-8') as f:
            json.dump([n.model_dump() for n in new_layer_nodes], f, indent=2, ensure_ascii=False)
            
        # Cập nhật cho vòng lặp tiếp theo
        current_layer_nodes = new_layer_nodes

    logger.info(f"--- Finished processing for User {user_id} ---\n")


def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("terminal_log.txt", mode='w', encoding='utf-8'),
        ]
    )

    print("*" * 80)
    print("* PIPELINE TEST *")
    print("*" * 80)

    # 1. Load dữ liệu từ Dataloader 
    test_data = run_dataloader_test()
    
    if not test_data:
        print("No test data found. Please ensure the dataloader is working correctly.")
        return

    # 2. Lặp qua từng Dataset và từng User
    for dataset_name, users_dict in test_data.items():
        for user_id, records in users_dict.items():
            if not records:
                print(f"Skipping User {user_id} ({dataset_name}) because no data is available.")
                continue
                
            # Tạo Logger và thư mục lưu trữ riêng cho user
            user_logger, user_output_dir = create_user_logger(dataset_name, user_id)
            
            # 3. Sử dụng Context Manager để cô lập Logger
            with user_logger_context(user_logger):
                logger = get_current_logger()
                try:
                    process_single_user(dataset_name, user_id, records, user_output_dir)
                except Exception as e:
                    logger.error(f"Crash pipeline at User {user_id}: {e}", exc_info=True)

    print("\n" + "*" * 80)
    print("* PIPELINE TEST HAS COMPLETED *")
    print("*" * 80)

if __name__ == "__main__":
    main()