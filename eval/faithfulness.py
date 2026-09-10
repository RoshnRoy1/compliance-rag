import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

llm = Groq(api_key=GROQ_API_KEY)

FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to the source context it was given.

Context provided to the AI:
{context}

AI's answer:
{answer}

Task: Check every factual claim in the AI's answer (ignore citation markers like "Source: X, Page: Y", those are references, not claims). For each substantive factual claim, determine if it is directly supported by the context above.

Respond with ONLY a JSON object in this exact format, no other text:
{{"faithful_claims": <number>, "total_claims": <number>, "unsupported_claims": ["<claim1>", "<claim2>"]}}

If all claims are supported, unsupported_claims should be an empty list."""


def score_faithfulness(question, answer, sources):
    context = "\n\n".join([s["text"] for s in sources])

    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    import json
    try:
        result = json.loads(raw)
        faithful = result.get("faithful_claims", 0)
        total = result.get("total_claims", 1)
        print(f"  DEBUG raw judge output: {result}")
        score = faithful / total if total > 0 else 0
        return {
            "score": round(score, 2),
            "faithful_claims": faithful,
            "total_claims": total,
            "unsupported_claims": result.get("unsupported_claims", []),
        }
    except json.JSONDecodeError:
        return {"score": None, "error": "Could not parse faithfulness response", "raw": raw}