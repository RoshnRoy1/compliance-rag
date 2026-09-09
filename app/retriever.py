import os
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from config import CHROMA_PATH, COLLECTION_NAME, DEFAULT_TOP_K, DOCS_FOLDER
from sentence_transformers import CrossEncoder

# Vector search setup
ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

# BM25 keyword search setup
# Load all chunks once at startup for keyword search
print("Loading chunks for BM25 index...")
all_data = collection.get(include=["documents", "metadatas"])
all_docs = all_data["documents"]
all_metadatas = all_data["metadatas"]
all_ids = all_data["ids"]

# Tokenize documents for BM25
tokenized_docs = [doc.lower().split() for doc in all_docs]
bm25_index = BM25Okapi(tokenized_docs)
print(f"BM25 index built over {len(all_docs)} chunks.")

# Cross-encoder reranker setup
print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker loaded.")


def get_chunk_count():
    return collection.count()


def get_available_docs():
    docs = []
    for f in sorted(os.listdir(DOCS_FOLDER)):
        if f.endswith(".pdf"):
            docs.append(f)
    return docs


def _format_result(doc, metadata, score, method):
    return {
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page", "?"),
        "distance": round(score, 4),
        "text": doc,
        "method": method,
    }


def _vector_search(question, top_k=10, where_filter=None):
    """Semantic search using embeddings."""
    kwargs = {
        "query_texts": [question],
        "n_results": top_k,
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    sources = []
    for i, doc in enumerate(results["documents"][0]):
        sources.append(_format_result(
            doc,
            results["metadatas"][0][i],
            results["distances"][0][i],
            "vector",
        ))
    return sources


def _bm25_search(question, top_k=10, doc_filter=None):
    """Keyword search using BM25."""
    tokenized_query = question.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    # Pair each score with its index
    scored = list(enumerate(scores))

    # Filter by document if needed
    if doc_filter:
        doc_path = "docs/" + doc_filter
        scored = [
            (i, s) for i, s in scored
            if all_metadatas[i].get("source") == doc_path
        ]

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    sources = []
    for i, score in top:
        if score > 0:
            # Convert BM25 score to a pseudo-distance (lower = better) for consistency
            sources.append(_format_result(
                all_docs[i],
                all_metadatas[i],
                1 / (1 + score),
                "keyword",
            ))
    return sources


def _merge_results(vector_results, bm25_results, top_k=3):
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion (RRF).
    
    RRF works by giving each result a score based on its rank position
    in each list, then combining scores. A result that appears in both
    lists gets a higher combined score. This is better than raw score
    merging because vector distances and BM25 scores are on completely
    different scales.
    """
    k = 60  # RRF constant, standard value from the original paper

    scores = {}  # key: chunk text, value: {"score": float, "result": dict}

    for rank, result in enumerate(vector_results):
        key = result["text"]
        rrf_score = 1 / (k + rank + 1)
        if key not in scores:
            scores[key] = {"score": 0, "result": result}
        scores[key]["score"] += rrf_score
        scores[key]["result"]["method"] = "vector"

    for rank, result in enumerate(bm25_results):
        key = result["text"]
        rrf_score = 1 / (k + rank + 1)
        if key not in scores:
            scores[key] = {"score": 0, "result": result}
        else:
            # This chunk appeared in both searches, mark it as hybrid
            scores[key]["result"]["method"] = "hybrid"
        scores[key]["score"] += rrf_score

    # Sort by combined RRF score (highest first)
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    # Take top_k and add the RRF score as the distance for display
    results = []
    for item in ranked[:top_k]:
        result = item["result"]
        result["rrf_score"] = round(item["score"], 4)
        results.append(result)

    return results

def _rerank(question, candidates, top_k=3):
    """
    Re-score candidates using a cross-encoder that reads the question
    and each chunk together, rather than comparing separate embeddings.
    This catches relevance that vector/BM25 similarity misses.
    """
    if not candidates:
        return []

    pairs = [[question, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def retrieve(question, top_k=DEFAULT_TOP_K):
    """Hybrid retrieval: vector + BM25, merged with RRF, then reranked."""
    vector_results = _vector_search(question, top_k=top_k * 5)
    bm25_results = _bm25_search(question, top_k=top_k * 5)
    fused = _merge_results(vector_results, bm25_results, top_k=top_k * 3)
    return _rerank(question, fused, top_k=top_k)


def retrieve_from_doc(question, doc_name, top_k=DEFAULT_TOP_K):
    """Hybrid retrieval filtered to one document, then reranked."""
    doc_path = "docs/" + doc_name
    vector_results = _vector_search(question, top_k=top_k * 5, where_filter={"source": doc_path})
    bm25_results = _bm25_search(question, top_k=top_k * 5, doc_filter=doc_name)
    fused = _merge_results(vector_results, bm25_results, top_k=top_k * 3)
    return _rerank(question, fused, top_k=top_k)