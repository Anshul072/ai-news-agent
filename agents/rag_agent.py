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

_SYSTEM_PROMPT = """You are an AI news analyst. Answer the user's question using ONLY the provided articles below.
Return ONLY valid JSON with keys:
- answer: your grounded answer as a string
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
        summary = enriched.get("summary", "") if enriched else ""
        context_parts.append(
            f"[Article {article_id}] {raw.get('title', '')} ({raw.get('source_name', '')})\n"
            f"Summary: {summary}"
        )
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
