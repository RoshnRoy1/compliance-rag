# Compliance RAG Assistant

**[Live Demo](https://compliance-rag.vercel.app/)** | **[Architecture](#architecture)** | **[Evaluation](#evaluation)**

A production-grade agentic RAG assistant over regulatory documents (GDPR, EU AI Act, UK Data Protection Act 2018) that grounds every answer in cited source text. Built to address the hallucination and traceability problems that block LLM adoption in regulated industries.

## What it does

Upload compliance documents or query the pre-loaded regulatory corpus. The system retrieves relevant passages, generates a grounded answer, and cites the exact document and page for every claim. If the context doesn't contain the answer, it says so.

## Architecture

User question goes through a React frontend to a FastAPI backend. An agentic router (LLM at temperature=0) classifies each query into one of four actions: DIRECT (answer without retrieval), SEARCH_ALL (hybrid search across all documents), SEARCH_DOC (filtered to one specific regulation), or COMPARE (dual retrieval with structured comparison).

Retrieval uses a three-stage pipeline. Stage 1: vector search and BM25 keyword search each return 15 candidates. Stage 2: Reciprocal Rank Fusion merges both lists into 9 unique candidates. Stage 3: a cross-encoder reranker reads each candidate paired with the question and selects the top 3 by real relevance.

The LLM generates an answer using only the retrieved context, with citations to the exact document and page. Source PDFs are accessible from the UI.

## Evaluation

Measured on a 10-question evaluation set spanning all three source documents:

| Metric | Score |
|--------|-------|
| Source retrieval accuracy | 100% |
| Faithfulness (LLM-as-judge) | 98.6% |

Faithfulness scoring checks every factual claim in the generated answer against the retrieved context. Unsupported claims are flagged. Methodology adapted from RAGAS.

### Embedding fine-tuning

Fine-tuned all-MiniLM-L6-v2 on 60 domain-specific triplets using TripletLoss. All similarity scores improved (average +0.09). Source accuracy for embedding-only search remained at 70% because 60 examples was insufficient to change rank ordering. The full hybrid + reranking pipeline compensates, achieving 100% regardless.

## Key design decisions

**RAG over fine-tuning:** Documents stay external, retrievable, and citable. Update by swapping a PDF, no retraining.

**Agentic routing:** One LLM call classifies queries into four actions. Prevents retrieval on irrelevant queries and prevents false grounding.

**Hybrid retrieval:** Semantic search finds meaning, BM25 finds exact terms. Neither alone is sufficient for regulatory text where "Article 33" is a lookup target, not a concept.

**Cross-encoder reranking:** Two-stage architecture. Fast retrieval for recall, slow cross-encoder for precision. Standard production pattern.

**Metadata filtering:** One collection with where filters, not separate collections per document. Simpler architecture, same selectivity.

## Tech stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI |
| Frontend | React (Vite) |
| Vector DB | ChromaDB |
| LLM | Groq (Qwen 27B) |
| Embeddings | all-MiniLM-L6-v2 |
| Keyword search | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Observability | Langfuse |
| Deployment | Render (backend), Vercel (frontend), Docker |

## Run locally

```bash
git clone https://github.com/RoshnRoy1/compliance-rag.git
cd compliance-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
python ingest.py
uvicorn api:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Or with Docker:

```bash
docker compose up --build
```

## Project structure

```
config.py              # Centralized settings
ingest.py              # Document loading, chunking, embedding pipeline
api.py                 # FastAPI endpoints (thin HTTP layer)
app/
  agent.py             # Agentic router + 4 handlers
  retriever.py         # Hybrid search + RRF + reranking
  generator.py         # LLM prompt templates + generation
  ingest_file.py       # Single-file upload ingestion
eval/
  dataset.py           # 10-question evaluation set
  faithfulness.py      # LLM-as-judge faithfulness scorer
  run_eval.py          # Evaluation runner
finetune/
  build_dataset.py     # Triplet generation from eval set
  expand_dataset.py    # Synthetic question generation for training
  train.py             # Embedding model fine-tuning
  benchmark.py         # Before/after comparison
frontend/              # React chat interface
docs/                  # Source PDFs (GDPR, EU AI Act, UK DPA)
```

## What I would do next

- Streaming responses for better UX
- Expand eval set to 50+ questions
- Fine-tune with 500+ triplets to improve embedding-only accuracy
- Migrate from Chroma to Qdrant for production scale
- Add authentication and rate limiting
- Implement conversation memory for follow-up questions

## Author

Roshan Roy, MSc AI, Heriot-Watt University
