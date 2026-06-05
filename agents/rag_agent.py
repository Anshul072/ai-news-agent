import json

from groq import Groq

import config
from tools.embedder import embed

_client = Groq(api_key=config.GROQ_API_KEY)
_MODEL = "llama-3.3-70b-versatile"

_NO_DATA = {
    "answer": "Not enough data in the knowledge base to answer this question.",
    "citations": [],
}

_HISTORY_WINDOW = 10

_SYSTEM_PROMPT = """You are an AI news analyst. Your job is to answer the user's question by reasoning over the provided articles.

Treat the articles as source material — do NOT restate or paraphrase them. Instead, construct an answer in your own words that:
- Explains what happened and the sequence of events
- Identifies who was involved and what they did
- Explains why it matters and what changed
- Directly addresses what the user asked

Return ONLY valid JSON with keys:
- answer: your synthesised answer as a string
- cited_ids: list of integer article IDs you referenced

Articles:
{context}

Return ONLY the JSON object, no markdown fences."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


_ENRICHED_FIELDS = [
    ("summary", "Summary"),
    ("whats_new", "What's new"),
    ("key_concepts", "Key concepts"),
    ("who_made_it", "Who made it"),
    ("use_cases", "Use cases"),
    ("importance_reasoning", "Why it matters"),
]


def _build_context_part(article_id: int, raw: dict, enriched: dict | None) -> str:
    lines = [f"[Article {article_id}] {raw.get('title', '')} ({raw.get('source_name', '')})"]
    for field, label in _ENRICHED_FIELDS:
        value = (enriched or {}).get(field, "")
        if isinstance(value, list):
            value = ", ".join(v for v in value if v)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _build_messages(query: str, context: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT.format(context=context)}]
    for entry in history[-_HISTORY_WINDOW:]:
        messages.append({"role": "user", "content": entry["question"]})
        messages.append({"role": "assistant", "content": entry["answer"]})
    messages.append({"role": "user", "content": query})
    return messages


def answer_query(query: str, sqlite_store, chroma_store, n_results: int = 5, history: list[dict] | None = None) -> dict:
    query_embedding = embed(query)
    chunks = chroma_store.search(query_embedding, n_results=n_results)

    if not chunks:
        return dict(_NO_DATA)

    article_ids = list(dict.fromkeys(c["article_id"] for c in chunks))

    context_parts = []
    citations = []
    for article_id in article_ids:
        raw = sqlite_store.get_raw_article(article_id)
        enriched = sqlite_store.get_enriched_article(article_id)
        if raw is None:
            continue
        context_parts.append(_build_context_part(article_id, raw, enriched))
        citations.append({
            "article_id": article_id,
            "title": raw.get("title", ""),
            "source_name": raw.get("source_name", ""),
            "url": raw.get("url", ""),
        })

    if not context_parts:
        return dict(_NO_DATA)

    context = "\n\n".join(context_parts)
    messages = _build_messages(query, context, history or [])
    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=messages,
        )
        result = json.loads(_strip_fences(response.choices[0].message.content))
        cited_ids = set(result.get("cited_ids", []))
        filtered = [c for c in citations if c["article_id"] in cited_ids] or citations
        return {"answer": result.get("answer", ""), "citations": filtered}
    except (json.JSONDecodeError, Exception):
        return {"answer": "Unable to generate answer.", "citations": citations}
