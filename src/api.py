import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from src.pipeline import (
    load_kb_documents,
    chunk_paragraph_based,
    build_chroma_index,
    retrieve_chunks,
    generate_citation_strict_answers
)

load_dotenv()

app = FastAPI(title="Mini-RAG Pipeline API")

# Add CORS Middleware to support development servers or external fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load KB documents and build index once during API startup
try:
    print("Initializing RAG index for API...")
    docs = load_kb_documents("kb")
    chunks = chunk_paragraph_based(docs)
    collection = build_chroma_index(chunks)
    print("RAG index initialized successfully.")
except Exception as e:
    print(f"Error building RAG index at startup: {e}")
    collection = None
    chunks = []

class QueryRequest(BaseModel):
    question: str

class ChunkDetail(BaseModel):
    rank: int
    chunk_id: str
    doc_title: str
    score: float
    chunk_text: str

class QueryResponse(BaseModel):
    answer_label: str
    answer: str
    citations: list[str]
    retrieved_chunks: list[ChunkDetail] = []

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    # Serve src/index.html from the local workspace
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        # Fallback if served from parent workspace root
        index_path = "src/index.html"
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard UI file (src/index.html) not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/metrics")
def get_metrics():
    # Read metrics directly from generated pipeline artifacts
    metrics = {}
    files_to_load = {
        "eval": "artifacts/eval.json",
        "chunking_comparison": "artifacts/chunking_comparison.json",
        "grounding_check": "artifacts/grounding_check.json",
        "answers": "artifacts/answers.json",
        "retrieval": "artifacts/retrieval.json"
    }
    
    for key, path in files_to_load.items():
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    metrics[key] = json.load(f)
            except Exception as e:
                metrics[key] = {"error": f"Failed to parse {path}: {str(e)}"}
        else:
            metrics[key] = {"error": f"File {path} not found. Please run the pipeline first."}
            
    return metrics

@app.post("/answer", response_model=QueryResponse)
def answer_question(request: QueryRequest):
    if collection is None:
        raise HTTPException(status_code=500, detail="RAG search index is not initialized.")
        
    groq_api_key = os.environ.get("GROK_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROK_API_KEY is not defined in the environment.")
        
    try:
        # Wrap query in expected list format
        query_item = {
            "query_id": "API_Q",
            "question": request.question
        }
        
        # Retrieve top 3 chunks
        retrieval = retrieve_chunks(collection, [query_item], chunks, top_n=3)
        
        # Generate answer using Groq
        answers, _ = generate_citation_strict_answers(retrieval, groq_api_key)
        
        if not answers:
            raise HTTPException(status_code=500, detail="Failed to generate answer.")
            
        result = answers[0]
        return QueryResponse(
            answer_label=result["answer_label"],
            answer=result["answer"],
            citations=result["citations"],
            retrieved_chunks=[
                ChunkDetail(
                    rank=c["rank"],
                    chunk_id=c["chunk_id"],
                    doc_title=c["doc_title"],
                    score=c["score"],
                    chunk_text=c["chunk_text"]
                )
                for c in retrieval[0]["top_k"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

