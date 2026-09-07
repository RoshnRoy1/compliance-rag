import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

#Setup

ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("compliance_docs", embedding_function=ef)
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

#FastAPI app
app = FastAPI(title="Compliance RAG API")

#Allow React front to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Req Shape
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/api/query")
async def ask(q: QueryRequest):
    #1 Retrieve
    results = collection.query(query_texts= [q.question], n_results=q.top_k)

    #Build context
    sources =[]
    context_parts= []
    for i,doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i].get("source","unknown")
        page = results["metadatas"][0][i].get("page","unknown")
        dist = results["distances"][0][i]
        context_parts.append(f"[Source: {source}, Page: {page}]\n{doc}")
        sources.append({"source": source, "page": page, "distance": round(dist, 4), "text": doc})
        
    context = "\n\n".join(context_parts)

    #Generate 
    prompt = f"""You are a compliance assistant. Answer the question using ONLY the context provided below.
Rules:
- If the context does not contain the answer, say "I cannot find this in the provided documents."
- Cite the source document and page number for every claim you make.
- Be precise and direct.

Context:
{context}

Question: {q.question}

Answer:"""

    response = llm.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )

    return {
        "question": q.question,
        "answer": response.choices[0].message.content,
        "sources": sources
    }

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "chunks": collection.count()}