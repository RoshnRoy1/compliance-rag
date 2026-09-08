from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from app.retriever import retrieve, retrieve_from_doc, get_available_docs

llm = Groq(api_key=GROQ_API_KEY)

ROUTER_PROMPT = """You are a routing assistant for a compliance document system containing: {available_docs}

Given a user's question, decide which action to take. Respond with ONLY one of these formats:

DIRECT
SEARCH_ALL
SEARCH_DOC: <filename>
COMPARE: <filename1> vs <filename2>

Rules:
- DIRECT: greetings, general knowledge, math, anything unrelated to compliance
- SEARCH_ALL: general compliance questions not targeting a specific regulation
- SEARCH_DOC: question clearly targets one specific regulation (e.g. "What does GDPR say about..." → SEARCH_DOC: gdpr.pdf)
- COMPARE: question asks to compare or contrast two regulations (e.g. "How do GDPR and UK DPA differ on..." → COMPARE: gdpr.pdf vs uk_data_protection.pdf)

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

COMPARE_PROMPT = """You are a compliance assistant. Compare the regulations using ONLY the context provided below.
Rules:
- Structure your answer with clear sections for each regulation.
- Cite the source document and page number for every claim.
- Highlight key differences and similarities.
- If the context lacks information for one side of the comparison, say so.

Context from {doc1}:
{context1}

Context from {doc2}:
{context2}

Question: {question}

Comparison:"""

DIRECT_PROMPT = """You are a helpful compliance assistant. The user's message does not require searching compliance documents. Respond naturally and briefly. If they greet you, greet them back and mention you can help with questions about GDPR, EU AI Act, and UK Data Protection Act.

User: {question}

Response:"""


def call_llm(prompt, temperature=None, max_tokens=None):
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content


def route(question):
    available = ", ".join(get_available_docs())
    prompt = ROUTER_PROMPT.format(question=question, available_docs=available)
    decision = call_llm(prompt, temperature=0, max_tokens=30).strip()

    if decision.startswith("COMPARE:"):
        parts = decision.replace("COMPARE:", "").strip()
        docs = [d.strip() for d in parts.split(" vs ")]
        if len(docs) == 2:
            return "COMPARE", docs
        return "SEARCH_ALL", []

    if decision.startswith("SEARCH_DOC:"):
        doc = decision.replace("SEARCH_DOC:", "").strip()
        return "SEARCH_DOC", [doc]

    if "DIRECT" in decision:
        return "DIRECT", []

    return "SEARCH_ALL", []


def build_context(sources):
    parts = []
    for s in sources:
        parts.append(f"[Source: {s['source']}, Page: {s['page']}]\n{s['text']}")
    return "\n\n".join(parts)


def handle_direct(question):
    answer = call_llm(DIRECT_PROMPT.format(question=question))
    return {
        "question": question,
        "answer": answer,
        "sources": [],
        "routed": "direct",
    }


def handle_search_all(question, top_k=3):
    sources = retrieve(question, top_k=top_k)
    context = build_context(sources)
    answer = call_llm(COMPLIANCE_PROMPT.format(context=context, question=question))
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "routed": "search_all",
    }


def handle_search_doc(question, doc_name, top_k=3):
    sources = retrieve_from_doc(question, doc_name, top_k=top_k)

    if not sources:
        return {
            "question": question,
            "answer": f"No relevant content found in {doc_name} for this question.",
            "sources": [],
            "routed": "search_doc:" + doc_name,
        }

    context = build_context(sources)
    answer = call_llm(COMPLIANCE_PROMPT.format(context=context, question=question))
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "routed": "search_doc:" + doc_name,
    }


def handle_compare(question, doc1, doc2, top_k=3):
    sources1 = retrieve_from_doc(question, doc1, top_k=top_k)
    sources2 = retrieve_from_doc(question, doc2, top_k=top_k)

    context1 = build_context(sources1) if sources1 else "No relevant content found."
    context2 = build_context(sources2) if sources2 else "No relevant content found."

    answer = call_llm(COMPARE_PROMPT.format(
        doc1=doc1, doc2=doc2,
        context1=context1, context2=context2,
        question=question,
    ))

    return {
        "question": question,
        "answer": answer,
        "sources": sources1 + sources2,
        "routed": f"compare:{doc1} vs {doc2}",
    }


def ask(question, top_k=3):
    action, params = route(question)

    if action == "DIRECT":
        return handle_direct(question)
    elif action == "SEARCH_DOC" and params:
        return handle_search_doc(question, params[0], top_k=top_k)
    elif action == "COMPARE" and len(params) == 2:
        return handle_compare(question, params[0], params[1], top_k=top_k)
    else:
        return handle_search_all(question, top_k=top_k)