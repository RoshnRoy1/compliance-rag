import sys, os, json, random, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL
from app.retriever import collection

llm = Groq(api_key=GROQ_API_KEY)

GENERATE_PROMPT = """Given this excerpt from a regulatory document, generate exactly 1 specific factual question that this text answers. The question should be the kind a compliance officer or lawyer would ask.

Text:
{chunk}

Respond with ONLY the question, nothing else."""


def generate_question(chunk_text):
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": GENERATE_PROMPT.format(chunk=chunk_text)}],
        temperature=0.3,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def build_expanded_dataset(target_count=50):
    # Load all chunks
    all_data = collection.get(limit=5636, include=["documents", "metadatas"])
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    # Filter out very short or noisy chunks
    good_indices = [
        i for i, doc in enumerate(all_docs)
        if len(doc) > 150 and not doc.startswith("Commencement") and "insert" not in doc[:50]
    ]

    # Sample random chunks to generate questions from
    sampled = random.sample(good_indices, min(target_count, len(good_indices)))

    triplets = []
    for count, idx in enumerate(sampled):
        positive_text = all_docs[idx]

        try:
            question = generate_question(positive_text)
        except Exception as e:
            print(f"  Skipped chunk {idx}: {e}")
            continue

        # Pick a random negative from a different document
        pos_source = all_metas[idx].get("source", "")
        neg_candidates = [
            i for i in good_indices
            if all_metas[i].get("source", "") != pos_source
        ]
        neg_idx = random.choice(neg_candidates)
        negative_text = all_docs[neg_idx]

        triplets.append({
            "question": question,
            "positive": positive_text,
            "negative": negative_text,
            "source": pos_source,
        })

        print(f"[{count + 1}/{target_count}] Q: {question[:80]}...")

        # Rate limit: Groq free tier
        time.sleep(1)

    return triplets


if __name__ == "__main__":
    # Load existing triplets
    existing = []
    if os.path.exists("finetune/triplets.json"):
        with open("finetune/triplets.json") as f:
            existing = json.load(f)

    print(f"Existing triplets: {len(existing)}")
    print(f"Generating ~50 more...\n")

    new_triplets = build_expanded_dataset(target_count=50)

    combined = existing + new_triplets
    with open("finetune/triplets.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nTotal triplets: {len(combined)}")
    print("Saved to finetune/triplets.json")