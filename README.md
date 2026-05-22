# Replayable Mini-RAG Pipeline & Analyzer Dashboard

A robust, replayable mini-RAG (Retrieval-Augmented Generation) pipeline that ingests a local product knowledge base, indexes it using a custom local embedder in Chroma DB, answers user queries with citation-strict responses using Groq, and evaluates retrieval quality and citation grounding deterministically.

This project also includes a **visual frontend dashboard** for interactive querying, metrics visualization, and chunking strategy comparisons.

---

## 🛠️ Features & Architecture

### 1. Ingestion & Preprocessing
* **Header Parsing**: Preprocesses documents from `kb/`, extracting `Title:` and `Section:` metadata lines and preserving raw text character offsets.
* **Deterministic Chunking**: Compares character-based Fixed-size chunking (Strategy A) with Paragraph/Line-based chunking (Strategy B) to analyze vocabulary fragmentation tradeoffs.

### 2. Local Indexing & Vector Search
* **Offline Embeddings**: Utilizes a custom pure-Python TF-IDF vectorizer (`SimpleTfidfVectorizer`) inside a custom Chroma DB embedding function, ensuring local, offline, and lightweight retrieval without heavy external model downloads.
* **Prepended Headers**: Prefixes chunk texts with Title/Section metadata to optimize query-context alignment and cosine similarity scoring.

### 3. Citation-Strict LLM Generation
* Generates answers using Groq's models (`llama-3.3-70b-versatile`) in strict JSON format.
* Output classification maps to controlled vocabularies: `grounded_answer`, `insufficient_context`, and `conflicting_context`.
* Integrates strict citation parsing with the format `[doc_title §chunk_id]`.
* Formally logs all LLM call records to `llm_calls.jsonl`.

### 4. Evaluation & Verification Suite
* **Retrieval Evaluation**: Deterministically evaluates if expected titles appear in the top-3 results (`hit`, `partial_hit`, or `miss`) and logs metrics to `artifacts/eval.json`.
* **Grounding Check**: Verifies that every cited chunk was actually retrieved and measures the text overlap ratio between the generated answer and the source document.
* **Pipeline Validation**: A command-line script (`validate.py`) asserts schema rules, score formats, citation integrity, and controlled vocabulary compliance.

### 5. Interactive Dashboard Frontend
* Serves a glassmorphic dashboard UI at `http://127.0.0.1:8000/`.
* Includes an **Interactive Playground** to query the pipeline and inspect reference chunks, as well as visualizations for hit rates, baseline cases, chunking strategies, and grounding validation checks.

---

## 🚀 Setup & Execution

### 1. Requirements Setup
This codebase supports Python 3.10+. Install the required dependencies (preferably using `uv` for fast resolution):
```bash
# Using uv (Recommended)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROK_API_KEY=your-groq-api-key-here
```

### 3. Run the Pipeline
To ingest documents, build indices, query baseline questions, and generate the analysis reports in `artifacts/`:
```bash
make run
```
*(Or run `python src/pipeline.py`)*

### 4. Run Validation
To programmatically check that all requirements and file schemas are met:
```bash
make validate
```
*(Or run `python validate.py`)*

### 5. Start the Web Dashboard
To launch the FastAPI server and explore the web interface:
```bash
make serve
```
*(Or run `uvicorn src.api:app --host 127.0.0.1 --port 8000`)*

Then open your browser and navigate to: **`http://127.0.0.1:8000/`**

---

## 📂 Project Structure
```text
├── artifacts/                  # Generated evaluation & metrics reports
│   ├── chunks.json             # Extracted paragraphs chunks
│   ├── retrieval.json          # Top-3 retrieved contexts per query
│   ├── answers.json            # Grounded answers and citation lists
│   ├── eval.json               # Retrieval accuracy and hit rates
│   ├── grounding_check.json    # Deterministic citation overlap checks
│   └── chunking_comparison.json# Strategy A vs B tradeoffs analysis
├── kb/                         # Knowledge Base text files
├── src/
│   ├── api.py                  # FastAPI application server
│   ├── index.html              # HTML/JS/CSS dashboard interface
│   └── pipeline.py             # Main RAG execution state machine
├── .gitignore                  # Git untracked pattern definitions
├── Makefile                    # Target command shortcuts (run, validate, serve)
├── llm_calls.jsonl             # Log of Groq API completion requests
├── queries.json                # Evaluator queries & ground truths
├── requirements.txt            # Python dependencies configuration
└── validate.py                 # Automatic validation suite
```
