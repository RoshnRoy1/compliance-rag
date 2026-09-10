import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.dataset import EVAL_SET
from eval.faithfulness import score_faithfulness
from app.agent import ask

def run_eval():
    print(f"Running evaluation on {len(EVAL_SET)} questions...\n")

    results = []
    correct_source_count = 0
    faithfulness_scores = []

    for i, item in enumerate(EVAL_SET):
        question = item["question"]
        expected_source = item["expected_source"]

        response = ask(question)
        sources = response.get("sources", [])
        answer = response.get("answer", "")

        retrieved_sources = set()
        for s in sources:
            filename = s["source"].replace("docs/", "")
            retrieved_sources.add(filename)

        if expected_source == "both":
            source_correct = len(retrieved_sources) >= 2
        else:
            source_correct = expected_source in retrieved_sources

        if source_correct:
            correct_source_count += 1

        # Skip faithfulness check for direct answers (no sources to check against)
        faithfulness = None
        if sources:
            faithfulness = score_faithfulness(question, answer, sources)
            if faithfulness.get("score") is not None:
                faithfulness_scores.append(faithfulness["score"])

        results.append({
            "question": question,
            "source_correct": source_correct,
            "faithfulness": faithfulness,
            "routed": response.get("routed", "unknown"),
        })

        status = "PASS" if source_correct else "FAIL"
        faith_display = f"{faithfulness['score']*100:.0f}%" if faithfulness and faithfulness.get("score") is not None else "N/A"
        print(f"[{i+1}/{len(EVAL_SET)}] Source: {status} | Faithfulness: {faith_display} | routed: {response.get('routed')}")
        print(f"  Q: {question}")
        if faithfulness and faithfulness.get("unsupported_claims"):
            print(f"  Unsupported claims: {faithfulness['unsupported_claims']}")
        print()

    source_accuracy = correct_source_count / len(EVAL_SET) * 100
    avg_faithfulness = (sum(faithfulness_scores) / len(faithfulness_scores) * 100) if faithfulness_scores else 0

    print(f"{'='*55}")
    print(f"Source retrieval accuracy: {correct_source_count}/{len(EVAL_SET)} ({source_accuracy:.1f}%)")
    print(f"Average faithfulness score: {avg_faithfulness:.1f}%")
    print(f"{'='*55}")

    return results


if __name__ == "__main__":
    run_eval()