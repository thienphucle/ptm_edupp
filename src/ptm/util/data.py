import pandas as pd
import json
import os
from pathlib import Path

# Cấu hình đường dẫn (Khớp với dataloader.py của bạn)


PROJECT_ROOT = Path(__file__).resolve().parents[3] 
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "report_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_codestream_samples():
    print("Extracting CodeStream samples...")
    sub_path = DATA_DIR / "codestream" / "submissions.csv"
    prob_path = DATA_DIR / "codestream" / "problems.csv"
    
    df_sub = pd.read_csv(str(sub_path))
    df_prob = pd.read_csv(str(prob_path))
    
    # Lấy 2 user đầu tiên
    users = df_sub["user_id"].unique()[:2]
    df_sub = df_sub[df_sub["user_id"].isin(users)]
    
    # Merge
    df_merged = pd.merge(df_sub, df_prob, on="problem_id", how="left")
    
    samples = []
    # Lấy tối đa 5 rows mỗi user để báo cáo không bị quá dài
    for user in users:
        user_df = df_merged[df_merged["user_id"] == user].head(5)
        for _, row in user_df.iterrows():
            samples.append({
                "user_id": str(row["user_id"]),
                "problem_id": str(row["problem_id"]),
                "attempt_number": int(row["attempt_number"]),
                "language": str(row["programming_language"]),
                "final_verdict": str(row["final_verdict"]),
                "verdict_sequence_trace": str(row["verdict_sequence_trace"]),
                "source_code_snippet": str(row["source_code"]),
                "problem_description_snippet": str(row["problem_description"])
            })
            
    out_path = OUTPUT_DIR / "sample_codestream_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved CodeStream samples to {out_path}")

def extract_upsc2m_samples():
    print("Extracting UPSC2M samples...")
    att_path = DATA_DIR / "upsc2m" / "attempts.jsonl"
    q_path = DATA_DIR / "upsc2m" / "questions.json"
    
    df_att = pd.read_json(str(att_path), lines=True)
    
    with open(str(q_path), 'r', encoding='utf-8') as f:
        df_q = pd.DataFrame(json.load(f))
        
    df_att["question_id"] = df_att["question_id"].astype(str)
    df_q["id"] = df_q["id"].astype(str)
    
    # Lấy 2 user đầu tiên
    users = df_att["user_id"].unique()[:2]
    df_att = df_att[df_att["user_id"].isin(users)]
    
    # Merge
    df_merged = pd.merge(df_att, df_q, left_on="question_id", right_on="id", how="inner")
    
    samples = []
    for user in users:
        user_df = df_merged[df_merged["user_id"] == user].head(5)
        for _, row in user_df.iterrows():
            samples.append({
                "user_id": str(row["user_id"]),
                "question_id": str(row["question_id"]),
                "category": str(row["category"]),
                "question_snippet": str(row["question"]),
                "options": row["options"] if isinstance(row["options"], list) else [],
                "correct_answer_key": row["answer"],
                "user_answer_key": str(row["user_answer"]),
                "is_correct": bool(row["user_correct"]),
                "time_taken_sec": int(row["time_taken"]),
                "difficulty": float(row["difficulty"])
            })
            
    out_path = OUTPUT_DIR / "sample_upsc2m_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved UPSC2M samples to {out_path}")

if __name__ == "__main__":
    extract_codestream_samples()
    extract_upsc2m_samples()
    print("🎉 Done! Check the 'report_samples' folder.")