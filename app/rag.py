from app.retriever import retrieve
from app.generator import generate

def ask(question, top_k=3):
    sources = retrieve(question, top_k=top_k)
    answer = generate(question, sources)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }