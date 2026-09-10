import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from eval.dataset import EVAL_SET
from config import CHROMA_PATH, COLLECTION_NAME

# Load both models
print("Loading baseline model...")
baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading fine-tuned model...")
finetuned_model = SentenceTransformer("finetune/compliance-MiniLM")

# Load raw chunks from Chroma for direct comparison
ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
all_data = collection.get(limit=5636, include=["documents", "metadatas"])
all_docs = all_data["documents"]
all_metas = all_data["metadatas"]

def search_with_model(model, question, top_k=3):
    """Embed question with a specific model, compare against all chunk embeddings."""
    q_embedding = model.encode([question])[0]
    chunk_embeddings = model.encode(all_docs, show_progress_bar=False, batch_size=128)

    import numpy as np
    similarities = np.dot(chunk_embeddings, q_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_embedding)
    )

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "source": all_metas[idx].get("source", "unknown"),
            "score": round(float(similarities[idx]), 4),
            "text": all_docs[idx][:100],
        })
    return results


print("\nEncoding all chunks with both models (this takes a minute)...\n")

# Pre-encode chunks with both models
import numpy as np
print("  Encoding with baseline...")
baseline_embeddings = baseline_model.encode(all_docs, show_progress_bar=True, batch_size=128)
print("  Encoding with fine-tuned...")
finetuned_embeddings = finetuned_model.encode(all_docs, show_progress_bar=True, batch_size=128)


def search_precomputed(embeddings, model, question, top_k=3):
    q_emb = model.encode([question])[0]
    sims = np.dot(embeddings, q_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        source = all_metas[idx].get("source", "unknown").replace("docs/", "")
        results.append({
            "source": source,
            "score": round(float(sims[idx]), 4),
        })
    return results, [all_metas[i].get("source", "").replace("docs/", "") for i in top_indices]


print("\n" + "=" * 65)
print(f"{'QUESTION':<45} {'BASELINE':>9} {'FINETUNED':>9}")
print("=" * 65)

baseline_correct = 0
finetuned_correct = 0

for item in EVAL_SET:
    q = item["question"]
    expected = item["expected_source"]

    b_results, b_sources = search_precomputed(baseline_embeddings, baseline_model, q)
    f_results, f_sources = search_precomputed(finetuned_embeddings, finetuned_model, q)

    if expected == "both":
        b_hit = len(set(b_sources)) >= 2
        f_hit = len(set(f_sources)) >= 2
    else:
        b_hit = expected in b_sources
        f_hit = expected in f_sources

    if b_hit:
        baseline_correct += 1
    if f_hit:
        finetuned_correct += 1

    b_label = "PASS" if b_hit else "FAIL"
    f_label = "PASS" if f_hit else "FAIL"

    short_q = q[:43] + ".." if len(q) > 45 else q
    print(f"{short_q:<45} {b_label:>9} {f_label:>9}")

    # Show top score comparison
    b_top = b_results[0]["score"]
    f_top = f_results[0]["score"]
    diff = f_top - b_top
    arrow = "+" if diff > 0 else ""
    print(f"  {'Top score:':<43} {b_top:>9.4f} {f_top:>9.4f}  ({arrow}{diff:.4f})")

print("=" * 65)
b_pct = baseline_correct / len(EVAL_SET) * 100
f_pct = finetuned_correct / len(EVAL_SET) * 100
print(f"{'Source accuracy:':<45} {b_pct:>8.1f}% {f_pct:>8.1f}%")
print("=" * 65)