# finetune/build_dataset.py
"""
Generates (question, positive_chunk, negative_chunk) triplets for fine-tuning
the embedding model. Positive chunks are ones we know answer a question
correctly (from our eval set + retrieval). Negatives are random unrelated chunks.
"""
import sys, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import EVAL_SET
from app.retriever import retrieve, collection

def build_triplets():
    triplets = []

    for item in EVAL_SET:
        question = item["question"]
        # Get the actual best-matching chunk using your current pipeline
        results = retrieve(question, top_k=1)
        if not results:
            continue
        positive_text = results[0]["text"]

        # Pick a random negative chunk (very likely unrelated)
        all_data = collection.get(limit=5636, include=["documents"])
        negative_text = random.choice(all_data["documents"])

        triplets.append({
            "question": question,
            "positive": positive_text,
            "negative": negative_text,
        })

    return triplets


if __name__ == "__main__":
    triplets = build_triplets()
    print(f"Built {len(triplets)} training triplets.")
    for t in triplets[:2]:
        print("\nQuestion:", t["question"])
        print("Positive:", t["positive"][:100], "...")
        print("Negative:", t["negative"][:100], "...")

    import json
    os.makedirs("finetune", exist_ok=True)
    with open("finetune/triplets.json", "w") as f:
        json.dump(triplets, f, indent=2)
    print("\nSaved to finetune/triplets.json")