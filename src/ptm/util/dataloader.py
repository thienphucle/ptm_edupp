import os
import json
import pandas as pd
import numpy as np 
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import List, Dict, Any
from pathlib import Path


from .datastructure import (
    CodeStreamRecord, 
    UPSC2MRecord, 
#    FoundationalASSISTRecord
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

CONFIG = {
    "codestream": {
        "submissions": DATA_DIR / "codestream" / "submissions.csv",
        "problems": DATA_DIR / "codestream" / "problems.csv",
    },
    "upsc2m": {
        "attempts": DATA_DIR / "upsc2m" / "attempts.jsonl",
        "questions": DATA_DIR / "upsc2m" / "questions.json",
    },
}

NUM_USERS_TO_TEST = 1 

def read_arrow_auto(file_path):
    """Đọc file Arrow tự động (File hoặc Stream)"""
    with pa.memory_map(file_path, 'r') as source:
        try:
            reader = ipc.RecordBatchFileReader(source)
        except Exception:
            source.seek(0)
            reader = ipc.RecordBatchStreamReader(source)
        table = reader.read_all()
    return table.to_pandas()



def load_codestream(concepts_map: dict) -> Dict[str, List[CodeStreamRecord]]:
    print("Loading CodeStream (Test Mode: 1 User)...")
    df_sub = pd.read_csv(CONFIG["codestream"]["submissions"])
    df_prob = pd.read_csv(CONFIG["codestream"]["problems"])
    
    test_users = df_sub["user_id"].unique()[1:2] 
    print(f"  -> Selected Users: {test_users}")
    
    df_sub = df_sub[df_sub["user_id"].isin(test_users)]
    df_merged = pd.merge(df_sub, df_prob, on="problem_id", how="left")
    
    user_data = {str(u): [] for u in test_users}
    
    for _, row in df_merged.iterrows():
        try:
            row_dict = row.to_dict()
            prob_id = str(row_dict["problem_id"])
            
            # load concept from concepts_map, fallback to default if not found
            concept_data = concepts_map.get(prob_id, {})
            concept_str = f"{concept_data.get('paradigm', 'General')} - {concept_data.get('specific_technique', 'Algorithm')}: {concept_data.get('core_challenge', '')}"
            
            row_dict["problem_concept"] = concept_str
            
            # Remove the problem_description field to avoid validation errors
            row_dict.pop("problem_description", None)
            
            record = CodeStreamRecord(**row_dict)
            user_data[str(record.user_id)].append(record)
        except Exception as e:
            print(f"  [Validation Error] CodeStream: {e}")
            
    return user_data


def load_upsc2m() -> Dict[str, List[UPSC2MRecord]]:
    print("Loading UPSC2M (Test Mode: 1 User)...")
    df_att = pd.read_json(CONFIG["upsc2m"]["attempts"], lines=True)
    df_q = pd.read_json(CONFIG["upsc2m"]["questions"])
    
    test_users = df_att["user_id"].unique()[1:2] 
    print(f"  -> Selected Users: {test_users}")
    
    df_att = df_att[df_att["user_id"].isin(test_users)].copy()
    df_att["question_id"] = df_att["question_id"].astype(str)
    df_q["id"] = df_q["id"].astype(str)
    
    df_merged = pd.merge(df_att, df_q, left_on="question_id", right_on="id", how="inner")
    if "id" in df_merged.columns:
        df_merged = df_merged.drop(columns=["id"])
        
    user_data = {str(u): [] for u in test_users}
    
    for _, row in df_merged.iterrows():
        try:
            row_dict = row.to_dict()
            
            row_dict["question"] = str(row_dict["question"])
            row_dict["user_id"] = str(row_dict["user_id"])
            row_dict["question_id"] = str(row_dict["question_id"])
            row_dict["user_answer"] = str(row_dict["user_answer"])
            
            if isinstance(row_dict.get("options"), np.ndarray):
                row_dict["options"] = row_dict["options"].tolist()
                
            record = UPSC2MRecord(**row_dict)
            user_data[str(record.user_id)].append(record)
        except Exception as e:
            print(f"  [Validation Error] UPSC2M: {e}")
            
    return user_data



# ==========================================
# MAIN TEST FUNCTION
# ==========================================
def run_dataloader_test(concept_cache_path: str = "logs/codestream_problem_concepts.json"):
    """
    Hàm chạy test chính. Trả về cấu trúc:
    {
        "codestream": { "user_1": [Record, Record...], "user_2": [...] },
        "upsc2m": { ... },
        "foundational_assist": { ... }
    }
    """
    test_batch = {}
        
    

    try:
        from .llm_handler import extract_all_problem_concepts
        
        if not os.path.exists(concept_cache_path):
            print(f"  [Concept Extraction] Cache not found. Extracting concepts for CodeStream...")
            # Tạo thư mục logs nếu chưa có
            os.makedirs(os.path.dirname(concept_cache_path), exist_ok=True)
            
            df_prob = pd.read_csv(str(CONFIG["codestream"]["problems"]))
            concepts_map = extract_all_problem_concepts(df_prob, concept_cache_path)
        else:
            print(f"  [Cache] Loading problem concepts from {concept_cache_path}")
            with open(concept_cache_path, 'r', encoding='utf-8') as f:
                concepts_map = json.load(f)
                
        test_batch["codestream"] = load_codestream(concepts_map=concepts_map)
    except Exception as e:
        print(f"Error loading CodeStream: {e}")

    try:
        test_batch["upsc2m"] = load_upsc2m()
    except Exception as e:
        print(f"Error loading UPSC2M: {e}")
        
    for dataset_name, users_dict in test_batch.items():
        print(f"\nDataset: {dataset_name.upper()}")
        for user_id, records in users_dict.items():
            print(f"  User: {user_id} | L0 Records: {len(records)}")
            
    return test_batch

if __name__ == "__main__":
    data = run_dataloader_test()
