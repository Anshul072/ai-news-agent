import json

from google import genai

import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_PROMPT_TEMPLATE = """You are an AI news analyst. Given the article below, extract structured information and return ONLY valid JSON with these exact keys:

- summary: 2-3 sentence plain English summary
- whats_new: the specific advancement or claim
- key_concepts: list of 3-5 concept names (strings)
- concept_explanations: object mapping each concept name to a brief explanation with analogies
- who_made_it: organization or researchers behind the work
- use_cases: list of practical applications (strings)
- importance_score: integer 1-10
- importance_reasoning: justification for the score

Article title: {title}
Article content: {content}

Return ONLY the JSON object, no markdown fences."""


def parse_articles(articles: list[dict]) -> list[dict]:
    results = []
    for article in articles:
        prompt = _PROMPT_TEMPLATE.format(
            title=article.get("title", ""),
            content=article.get("content", ""),
        )
        try:
            response = _client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            enriched = json.loads(response.text)
            enriched["article_id"] = article["id"]
            results.append(enriched)
        except (json.JSONDecodeError, KeyError, Exception):
            continue
    return results
