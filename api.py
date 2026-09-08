import os
import glob
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config import DOCS_FOLDER
from app.agent import ask
from app.retriever import get_chunk_count


app = FastAPI(title="Compliance RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/api/query")
async def query(q: QueryRequest):
    return ask(q.question, top_k=q.top_k)

@app.get("/api/health")
async def health():
    return {"status": "ok", "chunks": get_chunk_count()}

@app.get("/api/documents")
async def list_documents():
    docs = []
    for path in sorted(glob.glob(DOCS_FOLDER + "/*.pdf")):
        filename = os.path.basename(path)
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
        docs.append({"filename": filename, "path": path, "size_mb": size_mb})
    return {"documents": docs}

@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    path = os.path.join(DOCS_FOLDER, filename)
    if not os.path.exists(path):
        return {"error": "Document not found"}
    return FileResponse(path, media_type="application/pdf", filename=filename)