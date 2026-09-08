import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_PATH, COLLECTION_NAME, DEFAULT_TOP_K

ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

def get_chunk_count():
    return collection.count()

def retrieve(question, top_k=DEFAULT_TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)

    sources = []
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i].get("source", "unknown")
        page = results["metadatas"][0][i].get("page", "?")
        dist = results["distances"][0][i]
        sources.append({
            "source": source,
            "page": page,
            "distance": round(dist, 4),
            "text": doc,
        })

    return sources