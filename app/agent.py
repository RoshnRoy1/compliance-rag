from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from app.retriever import retrieve

llm = Groq(api_key=GROQ_API_KEY)

ROUTER_PROMPT = """You are a routing assistant. Your job is to decide whether a user's question requires searching compliance documents (GDPR, EU AI Act, UK Data Protection Act) or can be answered directly.

Respond with ONLY one word:
- "RETRIEVE" if the question is about regulations, compliance, data protection, AI governance, legal requirements, penalties, rights, or obligations
- "DIRECT" if the question is a greeting, general knowledge, math, or anything unrelated to compliance

Question: {question}

Decision:"""

COMPLIANCE_PROMPT = """You are a compliance assistant. Answer the question using ONLY the context provided below.
Rules:
- If the context does not contain the answer, say "I cannot find this in the provided documents."
- Cite the source document and page number for every claim you make.
- Be precise and direct.

Context:
{context}

Question: {question}

Answer:"""

DIRECT_PROMPT = """You are a helpful compliance assistant. The user's message does not require searching compliance documents. Respond naturally and briefly. If they greet you, greet them back and mention you can help with questions about GDPR, EU AI Act, and UK Data Protection Act.

User: {question}

Response:"""


def route(question):
    """Ask the LLM whether this question needs document retrieval."""
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=10,
    )
    decision = response.choices[0].message.content.strip().upper()
    return "RETRIEVE" if "RETRIEVE" in decision else "DIRECT"


def handle_direct(question):
    """Answer without retrieval."""
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": DIRECT_PROMPT.format(question=question)}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [],
        "routed": "direct",
    }


def handle_retrieve(question, top_k=3):
    """Answer with retrieval."""
    sources = retrieve(question, top_k=top_k)

    context_parts = []
    for s in sources:
        context_parts.append(f"[Source: {s['source']}, Page: {s['page']}]\n{s['text']}")
    context = "\n\n".join(context_parts)

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": COMPLIANCE_PROMPT.format(
            context=context, question=question
        )}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": sources,
        "routed": "retrieve",
    }


def ask(question, top_k=3):
    """Main entry point: route then handle."""
    decision = route(question)

    if decision == "RETRIEVE":
        return handle_retrieve(question, top_k=top_k)
    else:
        return handle_direct(question)