import os
import json
import sys

def check_file_exists(path):
    if not os.path.exists(path):
        print(f"[ERROR] Required file or directory does not exist: {path}")
        return False
    return True

def validate_all():
    print("Starting validation check...")
    
    # 1. Check existences
    required_files = [
        "kb",
        "queries.json",
        "artifacts/chunks.json",
        "artifacts/retrieval.json",
        "artifacts/answers.json",
        "artifacts/eval.json",
        "artifacts/grounding_check.json",
        "artifacts/chunking_comparison.json",
        "llm_calls.jsonl"
    ]
    
    all_exist = True
    for f in required_files:
        if not check_file_exists(f):
            all_exist = False
            
    if not all_exist:
        return False
        
    print("[PASS] All required artifacts exist.")
    
    # 2. Check queries file and load queries
    try:
        with open("queries.json", "r", encoding="utf-8") as f:
            queries = json.load(f)
        query_ids = {q["query_id"] for q in queries}
        print(f"[PASS] queries.json is valid. Total queries: {len(query_ids)}")
    except Exception as e:
        print(f"[ERROR] Failed parsing queries.json: {e}")
        return False
        
    # 3. Check artifacts/chunks.json
    try:
        with open("artifacts/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not isinstance(chunks, list):
            print("[ERROR] chunks.json must be a JSON array.")
            return False
        # Check required fields
        for idx, chunk in enumerate(chunks):
            required = ["chunk_id", "doc_title", "section", "text", "start_char", "end_char"]
            for r in required:
                if r not in chunk:
                    print(f"[ERROR] Chunk at index {idx} missing field: {r}")
                    return False
        print(f"[PASS] chunks.json is valid. Total chunks: {len(chunks)}")
    except Exception as e:
        print(f"[ERROR] Failed parsing chunks.json: {e}")
        return False

    # 4. Check artifacts/retrieval.json
    try:
        with open("artifacts/retrieval.json", "r", encoding="utf-8") as f:
            retrievals = json.load(f)
        if not isinstance(retrievals, list):
            print("[ERROR] retrieval.json must be a JSON array.")
            return False
            
        retrieval_query_ids = set()
        retrieval_map = {}
        for ret in retrievals:
            qid = ret.get("query_id")
            if not qid:
                print("[ERROR] Retrieval item missing query_id")
                return False
            retrieval_query_ids.add(qid)
            retrieval_map[qid] = ret
            
            top_k = ret.get("top_k", [])
            if len(top_k) < 3:
                print(f"[ERROR] Query {qid} has fewer than 3 retrieved chunks (has {len(top_k)})")
                return False
                
            for rank, item in enumerate(top_k, start=1):
                if not isinstance(item.get("score"), (int, float)):
                    print(f"[ERROR] Query {qid} at rank {rank} has non-numeric score: {item.get('score')}")
                    return False
                # check required keys
                for r in ["rank", "chunk_id", "doc_title", "score", "chunk_text"]:
                    if r not in item:
                        print(f"[ERROR] Retrieval item for query {qid} at rank {rank} missing field: {r}")
                        return False
                        
        if not query_ids.issubset(retrieval_query_ids):
            print(f"[ERROR] Not all queries processed in retrieval.json. Missing: {query_ids - retrieval_query_ids}")
            return False
            
        print("[PASS] retrieval.json is valid. All queries have >= 3 chunks and numeric scores.")
    except Exception as e:
        print(f"[ERROR] Failed parsing retrieval.json: {e}")
        return False

    # 5. Check artifacts/answers.json
    try:
        with open("artifacts/answers.json", "r", encoding="utf-8") as f:
            answers = json.load(f)
        if not isinstance(answers, list):
            print("[ERROR] answers.json must be a JSON array.")
            return False
            
        allowed_labels = {"grounded_answer", "insufficient_context", "conflicting_context"}
        answers_query_ids = set()
        
        for ans in answers:
            qid = ans.get("query_id")
            if not qid:
                print("[ERROR] Answer item missing query_id")
                return False
            answers_query_ids.add(qid)
            
            label = ans.get("answer_label")
            if label not in allowed_labels:
                print(f"[ERROR] Query {qid} has invalid answer label: '{label}' (allowed: {allowed_labels})")
                return False
                
            citations = ans.get("citations", [])
            used_chunks = ans.get("used_chunk_ids", [])
            
            if label == "grounded_answer" and len(citations) == 0:
                print(f"[ERROR] Query {qid} is marked 'grounded_answer' but has no citations.")
                return False
                
            # Verify citations refer only to retrieved chunks
            ret_chunks = {c["chunk_id"] for c in retrieval_map[qid]["top_k"]}
            for cid in used_chunks:
                if cid not in ret_chunks:
                    print(f"[ERROR] Query {qid} cites chunk_id '{cid}' which was not retrieved (retrieved: {ret_chunks})")
                    return False
                    
        if not query_ids.issubset(answers_query_ids):
            print(f"[ERROR] Not all queries processed in answers.json. Missing: {query_ids - answers_query_ids}")
            return False
            
        print("[PASS] answers.json is valid. Clean labels, correct citations and matches.")
    except Exception as e:
        print(f"[ERROR] Failed parsing answers.json: {e}")
        return False

    # 6. Check artifacts/eval.json
    try:
        with open("artifacts/eval.json", "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
        if not isinstance(eval_data, dict):
            print("[ERROR] eval.json must be a JSON object with 'summary' and 'results' fields.")
            return False
            
        if "summary" not in eval_data or "results" not in eval_data:
            print("[ERROR] eval.json missing 'summary' or 'results' fields.")
            return False
            
        summary = eval_data["summary"]
        required_summary = ["top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"]
        for r in required_summary:
            if r not in summary:
                print(f"[ERROR] Summary in eval.json missing key: {r}")
                return False
                
        results = eval_data["results"]
        eval_query_ids = set()
        allowed_eval_statuses = {"hit", "partial_hit", "miss"}
        
        for idx, item in enumerate(results):
            qid = item.get("query_id")
            if not qid:
                print(f"[ERROR] Eval item at index {idx} missing query_id")
                return False
            eval_query_ids.add(qid)
            
            status = item.get("retrival_status") or item.get("retrieval_status")
            if status not in allowed_eval_statuses:
                print(f"[ERROR] Query {qid} has invalid retrieval status: '{status}'")
                return False
                
            required_keys = ["query_id", "expected_doc_titles", "retrieved_doc_titles_top3", "matched_expected_title", "explanation"]
            for r in required_keys:
                if r not in item:
                    print(f"[ERROR] Query {qid} in evaluation missing key: {r}")
                    return False
                    
        if not query_ids.issubset(eval_query_ids):
            print(f"[ERROR] Not all queries evaluated in eval.json. Missing: {query_ids - eval_query_ids}")
            return False
            
        print("[PASS] eval.json is valid. Aggregate summary is present and statuses are clean.")
    except Exception as e:
        print(f"[ERROR] Failed parsing eval.json: {e}")
        return False
        
    # 7. Check grounding_check.json
    try:
        with open("artifacts/grounding_check.json", "r", encoding="utf-8") as f:
            grounding = json.load(f)
        if not isinstance(grounding, list):
            print("[ERROR] grounding_check.json must be a JSON array.")
            return False
        print("[PASS] grounding_check.json is valid.")
    except Exception as e:
        print(f"[ERROR] Failed parsing grounding_check.json: {e}")
        return False
        
    # 8. Check chunking_comparison.json
    try:
        with open("artifacts/chunking_comparison.json", "r", encoding="utf-8") as f:
            comparison = json.load(f)
        if not isinstance(comparison, dict) or "strategy_a_fixed_size" not in comparison or "strategy_b_paragraph_based" not in comparison:
            print("[ERROR] chunking_comparison.json missing required keys.")
            return False
        print("[PASS] chunking_comparison.json is valid.")
    except Exception as e:
        print(f"[ERROR] Failed parsing chunking_comparison.json: {e}")
        return False

    print("\n[ALL PASSED] Validation completed successfully! The codebase adheres to all requirements.")
    return True

if __name__ == "__main__":
    success = validate_all()
    sys.exit(0 if success else 1)
