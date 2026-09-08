from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

llm = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a compliance assistant. Answer the question using ONLY the context provided below.
Rules:
- If the context does not contain the answer, say "I cannot find this in the provided documents."
- Cite the source document and page number for every claim you make.
- Be precise and direct."""

def generate(question, sources):
    context_parts = []
    for s in sources:
        context_parts.append(f"[Source: {s['source']}, Page: {s['page']}]\n{s['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return response.choices[0].message.content