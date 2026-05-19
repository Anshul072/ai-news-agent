import json
import logging
import time

from groq import Groq
from json_repair import repair_json

import config

logger = logging.getLogger(__name__)
_client = Groq(api_key=config.GROQ_API_KEY)
_MODEL = "llama-3.3-70b-versatile"

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


_MIN_CONTENT_LENGTH = 80

# Expected type and default for each required output field.
_REQUIRED_FIELDS: dict[str, tuple] = {
    "summary":              (str,  ""),
    "whats_new":            (str,  ""),
    "key_concepts":         (list, []),
    "concept_explanations": (dict, {}),
    "who_made_it":          (str,  ""),
    "use_cases":            (list, []),
    "importance_score":     (int,  5),
    "importance_reasoning": (str,  ""),
}


def parse_articles(articles: list[dict]) -> list[dict]:
    results = []
    for article in articles:
        content = article.get("content", "")
        if len(content) < _MIN_CONTENT_LENGTH:
            logger.warning("Skipping (content too short): %s", article.get("title", "?")[:60])
            continue
        prompt = _PROMPT_TEMPLATE.format(
            title=article.get("title", ""),
            content=content,
        )
        response_text = _call_groq(prompt)
        if response_text is None:
            continue
        cleaned = _strip_fences(response_text)
        enriched = _safe_parse(cleaned, article.get("title", "?")[:60])
        if enriched is not None:
            enriched["article_id"] = article["id"]
            results.append(enriched)
    return results


def _safe_parse(text: str, title_hint: str) -> dict | None:
    """Try every recovery strategy in order; return a validated dict or None."""
    # 1. Direct parse
    try:
        return _normalize(json.loads(text), title_hint)
    except json.JSONDecodeError:
        pass

    # 2. Extract the first {...} block (handles prose before/after JSON)
    extracted = _extract_json_object(text)
    if extracted:
        try:
            return _normalize(json.loads(extracted), title_hint)
        except json.JSONDecodeError:
            pass

    # 3. Attempt structural repair (handles unquoted values, trailing commas, etc.)
    try:
        repaired = repair_json(extracted or text, return_objects=True)
        if isinstance(repaired, (dict, list)):
            result = _normalize(repaired, title_hint)
            if result is not None:
                logger.info("JSON repaired for: %s", title_hint)
                return result
    except Exception:
        pass

    # 4. Distinguish truncated responses from outright garbage
    if _is_truncated(text):
        logger.warning("Truncated JSON (token limit?) for: %s", title_hint)
        return None

    logger.warning(
        "JSON parse failed for: %s — response: %.120s",
        title_hint,
        text,
    )
    return None


def _normalize(parsed: object, title_hint: str) -> dict | None:
    """Coerce parsed value to a validated dict, or return None."""
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict):
            logger.warning("Model returned JSON array; using first element for: %s", title_hint)
            parsed = parsed[0]
        else:
            logger.warning("Model returned empty/invalid JSON array for: %s", title_hint)
            return None
    if not isinstance(parsed, dict):
        logger.warning("Unexpected JSON type %s for: %s", type(parsed).__name__, title_hint)
        return None
    return _validate_and_coerce(parsed)


def _validate_and_coerce(data: dict) -> dict:
    """Fill missing keys with defaults and coerce values to expected types."""
    result = dict(data)
    for key, (expected_type, default) in _REQUIRED_FIELDS.items():
        val = result.get(key)
        if val is None:
            result[key] = list(default) if isinstance(default, list) else (
                dict(default) if isinstance(default, dict) else default
            )
            continue
        if expected_type is list:
            if isinstance(val, str):
                result[key] = [v.strip() for v in val.split(",") if v.strip()]
            elif not isinstance(val, list):
                result[key] = [str(val)] if val else []
        elif expected_type is dict:
            if not isinstance(val, dict):
                result[key] = {}
        elif expected_type is int:
            if not isinstance(val, int) or isinstance(val, bool):
                try:
                    result[key] = int(float(str(val)))
                except (ValueError, TypeError):
                    result[key] = default
            if key == "importance_score":
                result[key] = max(1, min(10, result[key]))
        elif expected_type is str:
            if not isinstance(val, str):
                result[key] = str(val) if val is not None else default
    return result


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence line (handles ```json, ```JSON, ``` etc.)
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else text[3:]
        # Drop closing fence, tolerating trailing whitespace
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def _is_truncated(text: str) -> bool:
    """Unclosed braces/brackets indicate the model hit a token limit mid-response."""
    opens = text.count("{") + text.count("[")
    closes = text.count("}") + text.count("]")
    return opens > closes


_INTER_REQUEST_DELAY = 2.1  # seconds — keeps throughput just under the 30 RPM free-tier limit


def _call_groq(prompt: str, max_retries: int = 3) -> str | None:
    time.sleep(_INTER_REQUEST_DELAY)
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                wait = 60 * (attempt + 1)
                logger.warning("Rate limited — waiting %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.warning("Groq call failed: %s", exc)
                return None
    logger.warning("Groq call exhausted retries")
    return None
