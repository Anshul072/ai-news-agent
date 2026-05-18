import json

from google import genai

import config
from tools.embedder import embed

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_NO_DATA = {
    "answer": "Not enough data in the knowledge base to answer this question.",
    "citations": [],
}

_RAG_PROMPT = """You are an AI news analyst. Answer the question below using ONLY the provided articles.
Return ONLY valid JSON with keys:
- answer: your grounded answer as a string
- cited_ids: list of integer article IDs you referenced

Question: {query}

Articles:
{context}

Return ONLY the JSON object, no markdown fences."""


def answer_query(query: str, sqlite_store, chroma_store, n_results: int = 5) -> dict:
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
    prompt = _RAG_PROMPT.format(query=query, context=context)
    try:
        response = _client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        result = json.loads(response.text)
        cited_ids = set(result.get("cited_ids", []))
        filtered = [c for c in citations if c["article_id"] in cited_ids] or citations
        return {"answer": result.get("answer", ""), "citations": filtered}
    except (json.JSONDecodeError, Exception):
        return {"answer": "Unable to generate answer.", "citations": citations}
