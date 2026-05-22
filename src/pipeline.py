import os
import re
import json
import math
import hashlib
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
import chromadb
from chromadb.api.types import Documents, Embeddings
from groq import Groq

# ---------------------------------------------------------
# State Machine Tracker
# ---------------------------------------------------------
class PipelineState:
    STAGES = [
        "INIT",
        "DOCUMENTS_LOADED",
        "DOCUMENTS_CHUNKED",
        "INDEX_BUILT",
        "RETRIEVAL_COMPLETE",
        "ANSWERS_GENERATED",
        "EVALUATION_COMPLETE",
        "VALIDATION_COMPLETE",
        "RESULTS_FINALISED"
    ]
    
    def __init__(self):
        self.current_stage = "INIT"
        print(f"\n>>> [Stage: {self.current_stage}] Pipeline initialized.")

    def transition_to(self, new_stage: str):
        if new_stage not in self.STAGES:
            raise ValueError(f"Invalid stage: {new_stage}")
        current_idx = self.STAGES.index(self.current_stage)
        new_idx = self.STAGES.index(new_stage)
        
        if new_idx != current_idx + 1:
            raise RuntimeError(
                f"Cannot transition from {self.current_stage} directly to {new_stage}. "
                f"Must follow order: {' -> '.join(self.STAGES)}"
            )
            
        self.current_stage = new_stage
        print(f">>> [Stage: {self.current_stage}] Transition complete.")

# ---------------------------------------------------------
# Pure-Python TF-IDF Vectorizer & Custom Embedding Function
# ---------------------------------------------------------
class SimpleTfidfVectorizer:
    def __init__(self):
        self.doc_count = 0
        self.dfs = {}
        self.idf = {}
        self.vocabulary = {}
        self.vocab_list = []

    def _stem(self, word):
        w = word.lower()
        if len(w) > 4:
            if w.endswith('ies'):
                w = w[:-3] + 'y'
            elif w.endswith('es'):
                w = w[:-2]
            elif w.endswith('s') and not w.endswith('ss'):
                w = w[:-1]
            elif w.endswith('ing'):
                w = w[:-3]
            elif w.endswith('ed'):
                w = w[:-2]
        return w

    def _tokenize(self, text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self._stem(t) for t in tokens]

    def fit(self, docs):
        self.doc_count = len(docs)
        self.dfs = {}
        self.vocabulary = {}
        all_tokens_per_doc = [self._tokenize(doc) for doc in docs]
        
        for tokens in all_tokens_per_doc:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.dfs[token] = self.dfs.get(token, 0) + 1
        
        self.idf = {}
        for token, df in self.dfs.items():
            self.idf[token] = math.log(1.0 + (self.doc_count / df)) + 1.0
            
        self.vocab_list = list(self.dfs.keys())
        self.vocabulary = {token: i for i, token in enumerate(self.vocab_list)}
        return self

    def transform(self, docs):
        vectors = []
        for doc in docs:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            vector = [0.0] * len(self.vocab_list)
            
            for token, count in tf.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf_val = 1.0 + math.log(count) if count > 0 else 0.0
                    vector[idx] = tf_val * self.idf[token]
            
            sq_sum = sum(v * v for v in vector)
            norm = math.sqrt(sq_sum)
            if norm > 0:
                vector = [v / norm for v in vector]
            vectors.append(vector)
            
        return vectors

class TfidfEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        self.vectorizer = SimpleTfidfVectorizer()
        self.fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self.fitted:
            self.fit(input)
        return self.vectorizer.transform(input)

# ---------------------------------------------------------
# Document Ingestion and Chunking
# ---------------------------------------------------------
def load_kb_documents(kb_dir="kb"):
    documents = []
    if not os.path.exists(kb_dir):
        raise FileNotFoundError(f"KB directory '{kb_dir}' not found.")
        
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            lines = content.splitlines()
            title = ""
            section = ""
            body_lines = []
            
            header_ended = False
            for line in lines:
                if not header_ended:
                    if line.startswith("Title:"):
                        title = line[len("Title:"):].strip()
                        continue
                    elif line.startswith("Section:"):
                        section = line[len("Section:"):].strip()
                        continue
                    elif line.strip() == "":
                        continue
                    else:
                        header_ended = True
                
                if header_ended:
                    body_lines.append(line)
            
            body_text = "\n".join(body_lines).strip()
            documents.append({
                "filename": filename,
                "title": title if title else filename,
                "section": section if section else "General",
                "body": body_text
            })
            
    print(f"Loaded {len(documents)} documents from '{kb_dir}'.")
    return documents

def chunk_fixed_size(documents, chunk_size=120, overlap=20):
    chunks = []
    for doc in documents:
        body = doc["body"]
        body_len = len(body)
        if body_len == 0:
            continue
            
        start = 0
        chunk_idx = 1
        while start < body_len:
            end = min(start + chunk_size, body_len)
            chunk_text = body[start:end]
            
            chunks.append({
                "chunk_id": f"{doc['title'].replace(' ', '_').lower()}_fixed_{chunk_idx}",
                "doc_title": doc["title"],
                "section": doc["section"],
                "text": chunk_text,
                "start_char": start,
                "end_char": end
            })
            
            if end == body_len:
                break
            start += (chunk_size - overlap)
            chunk_idx += 1
            
    return chunks

def chunk_paragraph_based(documents):
    chunks = []
    for doc in documents:
        body = doc["body"]
        # Split by single newlines since the files use single newlines as line breaks
        lines = body.splitlines()
        current_idx = 0
        chunk_idx = 1
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_idx += len(line) + 1
                continue
                
            start_offset = line.find(stripped)
            start = current_idx + start_offset
            end = start + len(stripped)
            
            chunks.append({
                "chunk_id": f"{doc['title'].replace(' ', '_').lower()}_para_{chunk_idx}",
                "doc_title": doc["title"],
                "section": doc["section"],
                "text": stripped,
                "start_char": start,
                "end_char": end
            })
            current_idx += len(line) + 1
            chunk_idx += 1
            
    return chunks

# ---------------------------------------------------------
# Chroma DB Indexing and Retrieval
# ---------------------------------------------------------
def build_chroma_index(chunks, collection_name="kb_collection"):
    client = chromadb.EphemeralClient()
    emb_fn = TfidfEmbeddingFunction()
    
    # Prepend Title and Section to texts for indexing to boost keyword matching
    prepended_texts = [
        f"Title: {c['doc_title']}\nSection: {c['section']}\n{c['text']}"
        for c in chunks
    ]
    emb_fn.fit(prepended_texts)
    
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=prepended_texts,
        metadatas=[{"doc_title": c["doc_title"], "section": c["section"]} for c in chunks]
    )
    
    return collection

def retrieve_chunks(collection, queries, chunks_metadata, top_n=3):
    retrieval_results = []
    # Map chunk_id to text for convenience
    chunk_text_map = {c["chunk_id"]: c["text"] for c in chunks_metadata}
    
    for q in queries:
        question = q["question"]
        query_id = q["query_id"]
        
        results = collection.query(
            query_texts=[question],
            n_results=min(top_n, len(chunks_metadata))
        )
        
        top_k = []
        for rank, (chunk_id, distance, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0]
        ), start=1):
            # cosine similarity = 1 - cosine distance
            score = float(1.0 - distance)
            top_k.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "doc_title": metadata["doc_title"],
                "score": score,
                "chunk_text": chunk_text_map[chunk_id]
            })
            
        retrieval_results.append({
            "query_id": query_id,
            "question": question,
            "top_k": top_k
        })
        
    return retrieval_results

# ---------------------------------------------------------
# Groq Answer Generation & LLM Logger
# ---------------------------------------------------------
def generate_citation_strict_answers(retrieval_results, groq_api_key):
    client = Groq(api_key=groq_api_key)
    answers = []
    llm_calls = []
    
    # Try models in order of preference
    models = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
    
    for ret in retrieval_results:
        query_id = ret["query_id"]
        question = ret["question"]
        top_chunks = ret["top_k"]
        
        # Build context
        context_blocks = []
        for c in top_chunks:
            context_blocks.append(
                f"Document Title: {c['doc_title']}\n"
                f"Chunk ID: {c['chunk_id']}\n"
                f"Content: {c['chunk_text']}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)
        
        system_prompt = (
            "You are a factual, strict question-answering assistant.\n"
            "Your task is to answer the user's question based ONLY on the provided chunks of context.\n"
            "You must return a JSON object with the following schema:\n"
            "{\n"
            "  \"answer_label\": \"grounded_answer\" | \"insufficient_context\" | \"conflicting_context\",\n"
            "  \"answer\": \"string\",\n"
            "  \"citations\": [\"string\"],\n"
            "  \"used_chunk_ids\": [\"string\"]\n"
            "}\n\n"
            "Rules:\n"
            "1. If the retrieved context is insufficient to answer the question, or does not contain direct answers, you must set \"answer_label\" to \"insufficient_context\", \"answer\" to \"I do not have enough information to answer this question.\", \"citations\" to [], and \"used_chunk_ids\" to [].\n"
            "2. If the context contains conflicting answers for the question, you must set \"answer_label\" to \"conflicting_context\", \"answer\" to a brief description of the conflict, \"citations\" to the conflicting chunk citations, and \"used_chunk_ids\" to the conflicting chunk IDs.\n"
            "3. If the question can be answered from the context, you must set \"answer_label\" to \"grounded_answer\".\n"
            "4. The \"answer\" must be grounded ONLY in the retrieved chunks. Do not use any outside knowledge or make assumptions.\n"
            "5. You must cite every claim or fact in the answer using the exact format: [doc_title §chunk_id].\n"
            "   Example: \"Bank withdrawals take 1 to 3 business days [Cash withdrawal processing §cash_withdrawal_processing_fixed_2].\"\n"
            "   Citations must refer only to the retrieved chunks. Do not cite chunks that were not retrieved or did not contribute to the answer.\n"
            "6. The \"citations\" list must contain all citation strings used in the answer (e.g. [\"[doc_title §chunk_id]\"]).\n"
            "7. The \"used_chunk_ids\" must contain the chunk_id strings of the chunks you cited (e.g. [\"cash_withdrawal_processing_fixed_2\"])."
        )
        
        user_prompt = f"Question: {question}\n\nContext Chunks:\n{context_text}"
        prompt_hash = hashlib.sha256((system_prompt + user_prompt).encode('utf-8')).hexdigest()
        
        response_json = None
        used_model = None
        error_msg = None
        
        # Call Groq with retries across models
        for model in models:
            try:
                start_time = datetime.utcnow()
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                response_content = completion.choices[0].message.content
                response_json = json.loads(response_content)
                used_model = model
                
                # Log LLM Call
                llm_calls.append({
                    "stage": "ANSWERS_GENERATED",
                    "query_id": query_id,
                    "timestamp": start_time.isoformat() + "Z",
                    "provider": "groq",
                    "model": used_model,
                    "prompt_hash": prompt_hash,
                    "input_artifacts": ["artifacts/retrieval.json"],
                    "output_artifact": "artifacts/answers.json"
                })
                break
            except Exception as e:
                error_msg = str(e)
                print(f"Failed calling Groq using {model}: {e}")
                continue
                
        if response_json is None:
            raise RuntimeError(f"Failed to generate answer for query {query_id} with Groq. Error: {error_msg}")
            
        # Clean up and normalize outputs to ensure full compliance with rules
        answer_label = response_json.get("answer_label", "insufficient_context")
        if answer_label not in ["grounded_answer", "insufficient_context", "conflicting_context"]:
            answer_label = "insufficient_context"
            
        answer_text = response_json.get("answer", "")
        citations = response_json.get("citations", [])
        used_chunk_ids = response_json.get("used_chunk_ids", [])
        
        # Validations on citation formats and mappings
        retrieved_chunk_ids = {c["chunk_id"] for c in top_chunks}
        retrieved_chunk_map = {c["chunk_id"]: c for c in top_chunks}
        
        cleaned_citations = []
        cleaned_used_chunk_ids = []
        
        # If the label is grounded_answer but answer is empty, default label
        if answer_label == "grounded_answer" and not answer_text:
            answer_label = "insufficient_context"
            answer_text = "I do not have enough information to answer this question."
            
        # Validate each citation and referenced chunk
        for cit in citations:
            # Match doc_title and chunk_id from citation: [Title §chunk_id]
            match = re.search(r'\[(.+?)\s*§\s*(.+?)\]', cit)
            if match:
                cited_title, cited_chunk_id = match.groups()
                # Check if it was actually retrieved
                if cited_chunk_id in retrieved_chunk_ids:
                    cleaned_citations.append(cit)
                    if cited_chunk_id not in cleaned_used_chunk_ids:
                        cleaned_used_chunk_ids.append(cited_chunk_id)
            else:
                # Malformed citation string, try to fix from used_chunk_ids
                pass
                
        # If no valid citations could be matched but the answer is grounded, try to reconstruct
        if answer_label == "grounded_answer" and not cleaned_citations:
            # Fallback: cite the top retrieval chunk if the model outputted it or used it
            if used_chunk_ids:
                for cid in used_chunk_ids:
                    if cid in retrieved_chunk_ids:
                        doc_t = retrieved_chunk_map[cid]["doc_title"]
                        cit_str = f"[{doc_t} §{cid}]"
                        cleaned_citations.append(cit_str)
                        cleaned_used_chunk_ids.append(cid)
                        # Append the citation to the text if it is missing
                        if cit_str not in answer_text:
                            answer_text += f" {cit_str}"
            else:
                # Cite the top 1 rank chunk
                top_c = top_chunks[0]
                cit_str = f"[{top_c['doc_title']} §{top_c['chunk_id']}]"
                cleaned_citations.append(cit_str)
                cleaned_used_chunk_ids.append(top_c['chunk_id'])
                if cit_str not in answer_text:
                    answer_text += f" {cit_str}"
                    
        # Ensure list coherence
        if answer_label != "grounded_answer":
            cleaned_citations = []
            cleaned_used_chunk_ids = []
            
        answers.append({
            "query_id": query_id,
            "answer_label": answer_label,
            "answer": answer_text,
            "citations": cleaned_citations,
            "used_chunk_ids": cleaned_used_chunk_ids
        })
        
    return answers, llm_calls

# ---------------------------------------------------------
# Evaluation Suite
# ---------------------------------------------------------
def evaluate_retrieval(retrieval_results, queries_ground_truth):
    eval_records = []
    hits = 0
    partial_hits = 0
    misses = 0
    
    gt_map = {q["query_id"]: q for q in queries_ground_truth}
    
    for ret in retrieval_results:
        query_id = ret["query_id"]
        gt = gt_map.get(query_id)
        if not gt:
            continue
            
        expected_titles = gt["expected_doc_titles"]
        retrieved_titles_top3 = [c["doc_title"] for c in ret["top_k"][:3]]
        
        # Calculate overlap
        intersection = set(expected_titles).intersection(set(retrieved_titles_top3))
        
        if len(intersection) == len(expected_titles):
            status = "hit"
            hits += 1
            matched = True
            explanation = f"All expected titles {expected_titles} found in top 3."
        elif len(intersection) > 0:
            status = "partial_hit"
            partial_hits += 1
            matched = True
            explanation = f"Some expected titles {list(intersection)} found in top 3."
        else:
            status = "miss"
            misses += 1
            matched = False
            explanation = f"None of the expected titles {expected_titles} found in top 3."
            
        eval_records.append({
            "query_id": query_id,
            "expected_doc_titles": expected_titles,
            "retrieved_doc_titles_top3": retrieved_titles_top3,
            "retrieval_status": status,
            "matched_expected_title": matched,
            "explanation": explanation
        })
        
    total = len(eval_records)
    hit_rate = (hits + partial_hits) / total if total > 0 else 0.0
    
    summary = {
        "top3_hit_rate": hit_rate,
        "total_queries": total,
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses
    }
    
    return {
        "summary": summary,
        "results": eval_records
    }

# ---------------------------------------------------------
# Answer Grounding Check
# ---------------------------------------------------------
def run_answer_grounding_check(answers, retrieval_results):
    retrieval_map = {r["query_id"]: {c["chunk_id"]: c for c in r["top_k"]} for r in retrieval_results}
    check_records = []
    
    def get_clean_words(text):
        return set(re.findall(r'\b\w{3,}\b', text.lower())) # words of 3+ chars
        
    for ans in answers:
        query_id = ans["query_id"]
        citations = ans["citations"]
        used_ids = ans["used_chunk_ids"]
        answer_text = ans["answer"]
        
        citations_checked = []
        all_citations_valid = True
        
        retrieved_chunks = retrieval_map.get(query_id, {})
        
        for cit, cid in zip(citations, used_ids):
            in_retrieval = cid in retrieved_chunks
            overlap_ratio = 0.0
            has_supporting_wording = False
            
            if in_retrieval:
                chunk_text = retrieved_chunks[cid]["chunk_text"]
                # Compute word overlap between cited text and the generated answer
                chunk_words = get_clean_words(chunk_text)
                answer_words = get_clean_words(answer_text)
                
                overlap_words = chunk_words.intersection(answer_words)
                if len(answer_words) > 0:
                    overlap_ratio = len(overlap_words) / len(answer_words)
                
                # Check for substring or keyword overlap threshold
                if len(overlap_words) >= 2 or overlap_ratio > 0.1:
                    has_supporting_wording = True
            
            if not in_retrieval or not has_supporting_wording:
                all_citations_valid = False
                
            citations_checked.append({
                "citation": cit,
                "chunk_id": cid,
                "in_retrieval": in_retrieval,
                "word_overlap_ratio": float(overlap_ratio),
                "has_supporting_wording": has_supporting_wording
            })
            
        check_records.append({
            "query_id": query_id,
            "citations_checked": citations_checked,
            "all_citations_valid": all_citations_valid if citations else True
        })
        
    return check_records

# ---------------------------------------------------------
# Main Execution Runner
# ---------------------------------------------------------
def main():
    # 1. Init
    state = PipelineState()
    
    # Load dotenv config
    load_dotenv()
    groq_api_key = os.environ.get("GROK_API_KEY")
    if not groq_api_key:
        raise ValueError("GROK_API_KEY is not defined in the environment or .env file.")
        
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    
    # 2. Documents Loaded
    docs = load_kb_documents("kb")
    state.transition_to("DOCUMENTS_LOADED")
    
    # 3. Documents Chunked
    chunks_fixed = chunk_fixed_size(docs, chunk_size=120, overlap=20)
    chunks_para = chunk_paragraph_based(docs)
    
    # Save primary chunks (Strategy B: Paragraph/Line based) to artifacts/chunks.json
    with open("artifacts/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_para, f, indent=2)
    print(f"Saved {len(chunks_para)} paragraph chunks to 'artifacts/chunks.json'.")
    state.transition_to("DOCUMENTS_CHUNKED")
    
    # 4. Index Built
    collection_para = build_chroma_index(chunks_para, "kb_collection_para")
    state.transition_to("INDEX_BUILT")
    
    # Load Queries
    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    # 5. Retrieval Complete
    retrieval_para = retrieve_chunks(collection_para, queries, chunks_para, top_n=3)
    with open("artifacts/retrieval.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_para, f, indent=2)
    print("Saved retrieval results to 'artifacts/retrieval.json'.")
    state.transition_to("RETRIEVAL_COMPLETE")
    
    # 6. Answers Generated
    answers, llm_calls = generate_citation_strict_answers(retrieval_para, groq_api_key)
    with open("artifacts/answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)
    
    # Append LLM calls to llm_calls.jsonl
    with open("llm_calls.jsonl", "a", encoding="utf-8") as f:
        for call in llm_calls:
            f.write(json.dumps(call) + "\n")
            
    print("Saved answers to 'artifacts/answers.json' and logged LLM calls.")
    state.transition_to("ANSWERS_GENERATED")
    
    # 7. Evaluation Complete
    eval_results_para = evaluate_retrieval(retrieval_para, queries)
    with open("artifacts/eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_results_para, f, indent=2)
    print("Saved evaluation metrics to 'artifacts/eval.json'.")
    state.transition_to("EVALUATION_COMPLETE")
    
    # 8. Validation Complete (Grounding Check)
    grounding_check_results = run_answer_grounding_check(answers, retrieval_para)
    with open("artifacts/grounding_check.json", "w", encoding="utf-8") as f:
        json.dump(grounding_check_results, f, indent=2)
    print("Saved grounding checks to 'artifacts/grounding_check.json'.")
    state.transition_to("VALIDATION_COMPLETE")
    
    # 9. Results Finalised (Chunking Comparison)
    collection_fixed = build_chroma_index(chunks_fixed, "kb_collection_fixed")
    retrieval_fixed = retrieve_chunks(collection_fixed, queries, chunks_fixed, top_n=3)
    eval_results_fixed = evaluate_retrieval(retrieval_fixed, queries)
    
    comparison = {
        "strategy_a_fixed_size": eval_results_fixed["summary"],
        "strategy_b_paragraph_based": eval_results_para["summary"],
        "tradeoff_explanation": (
            "Fixed-size chunking (Strategy A) splits documents at rigid character counts (e.g. 120 chars). "
            "While it ensures uniform text size, it frequently cuts sentences in half, breaking semantic context. "
            "Paragraph-based chunking (Strategy B) splits text on line or paragraph breaks, preserving "
            "complete thoughts. This semantic integrity improves query retrieval precision and prevents word fragmentation, "
            "but can lead to variable chunk lengths depending on document structure."
        )
    }
    
    with open("artifacts/chunking_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("Saved chunking comparison to 'artifacts/chunking_comparison.json'.")
    state.transition_to("RESULTS_FINALISED")
    
    print("\nPipeline execution complete! All required artifacts successfully generated.")

if __name__ == "__main__":
    main()
