import json
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

_MODEL = "gemini-3.5-flash"
_FALLBACK_MODEL = "gemini-2.5-flash"
_RATE_LIMIT_SLEEP = 12  # seconds — keeps throughput at or below 5 RPM free tier

_client = None  # lazily initialised on first judge() call


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client

DIMENSIONS = {
    "news_parse_agent": [
        "summary_faithfulness",
        "importance_score_calibration",
        "key_concepts_relevance",
        "whats_new_specificity",
        "use_cases_plausibility",
    ],
    "sentiment_agent": [
        "label_score_consistency",
        "concern_use_case_groundedness",
        "quote_authenticity",
    ],
    "rag_agent": [
        "answer_relevance",
        "faithfulness",
        "citation_accuracy",
    ],
}


def _build_prompt(agent_name: str, inputs: dict, output: dict) -> str:
    output_str = json.dumps(output, indent=2)

    if agent_name == "news_parse_agent":
        return (
            "You are an LLM judge evaluating a news parse agent's output.\n"
            "Score each dimension 1-5 (integer). Provide a non-empty reasoning string for each.\n"
            "Return ONLY valid JSON — no markdown fences:\n\n"
            '{"summary_faithfulness": {"score": <int 1-5>, "reasoning": "<cite a specific sentence from the article>"},\n'
            ' "importance_score_calibration": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "key_concepts_relevance": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "whats_new_specificity": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "use_cases_plausibility": {"score": <int 1-5>, "reasoning": "<string>"}}\n\n'
            "Rubric:\n"
            "- summary_faithfulness: Does the summary accurately reflect the article without hallucinating? "
            "In reasoning, cite a specific sentence from the article that supports or contradicts the summary.\n"
            "- importance_score_calibration: Is the 1-10 importance_score appropriate (breakthrough=high, minor update=low)?\n"
            "- key_concepts_relevance: Are the key_concepts present and central in the article content?\n"
            "- whats_new_specificity: Does whats_new capture actual novelty, not just restate the summary?\n"
            "- use_cases_plausibility: Are use_cases grounded in the article, not generic AI boilerplate?\n\n"
            f"Article title: {inputs.get('article_title', '')}\n"
            f"Article content:\n{inputs.get('article_content', '')}\n\n"
            f"Agent output:\n{output_str}"
        )

    if agent_name == "sentiment_agent":
        return (
            "You are an LLM judge evaluating a sentiment analysis agent's output.\n"
            "Score each dimension 1-5 (integer). Provide a non-empty reasoning string for each.\n"
            "Return ONLY valid JSON — no markdown fences:\n\n"
            '{"label_score_consistency": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "concern_use_case_groundedness": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "quote_authenticity": {"score": <int 1-5>, "reasoning": "<string>"}}\n\n'
            "Rubric:\n"
            "- label_score_consistency: Does the sentiment_label (Positive/Negative/Mixed/Neutral) "
            "align with the numeric sentiment_score?\n"
            "- concern_use_case_groundedness: Are top_concerns and top_use_cases traceable to "
            "the HN thread comments provided?\n"
            "- quote_authenticity: Do notable_quotes read like real HN comments, not LLM-generated summaries?\n\n"
            f"Article title: {inputs.get('article_title', '')}\n"
            f"HN threads:\n{json.dumps(inputs.get('hn_threads', []), indent=2)}\n\n"
            f"Agent output:\n{output_str}"
        )

    if agent_name == "rag_agent":
        return (
            "You are an LLM judge evaluating a RAG agent's output.\n"
            "Score each dimension 1-5 (integer). Provide a non-empty reasoning string for each.\n"
            "Return ONLY valid JSON — no markdown fences:\n\n"
            '{"answer_relevance": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "faithfulness": {"score": <int 1-5>, "reasoning": "<string>"},\n'
            ' "citation_accuracy": {"score": <int 1-5>, "reasoning": "<string>"}}\n\n'
            "Rubric:\n"
            "- answer_relevance: Does the answer address what the query asked?\n"
            "- faithfulness: Is every claim in the answer supported by the retrieved context chunks? "
            "No external facts introduced?\n"
            "- citation_accuracy: Are the cited articles the ones that actually support the answer?\n\n"
            f"Query: {inputs.get('query', '')}\n"
            f"Retrieved context chunks:\n{inputs.get('context_chunks', '')}\n\n"
            f"Agent output:\n{output_str}"
        )

    raise ValueError(f"Unknown agent_name: {agent_name!r}")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _parse_response(text: str, agent_name: str) -> dict:
    dims = DIMENSIONS[agent_name]
    try:
        parsed = json.loads(_strip_fences(text))
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
    except (json.JSONDecodeError, ValueError):
        return {
            dim: {"score": 1, "reasoning": f"Judge returned malformed response: {text[:200]}"}
            for dim in dims
        }

    result = {}
    for dim in dims:
        entry = parsed.get(dim, {})
        if not isinstance(entry, dict):
            entry = {}
        try:
            score = max(1, min(5, int(entry.get("score", 1))))
        except (ValueError, TypeError):
            score = 1
        reasoning = entry.get("reasoning") or "No reasoning provided."
        result[dim] = {"score": score, "reasoning": str(reasoning)}

    return result


def judge(agent_name: str, inputs: dict, output: dict) -> dict:
    """Score agent output against a named rubric using Gemini Flash as judge.

    All dimensions for one article are scored in a single API call. A 12-second
    sleep is applied before each call to stay within the 5 RPM free-tier limit.
    If the primary model fails, retries once with the fallback model.

    Returns a dict mapping each dimension name to {"score": int, "reasoning": str}.
    Malformed API responses are handled gracefully — the function never raises.
    """
    if agent_name not in DIMENSIONS:
        raise ValueError(f"Unknown agent_name: {agent_name!r}. Must be one of: {list(DIMENSIONS)}")

    time.sleep(_RATE_LIMIT_SLEEP)

    prompt = _build_prompt(agent_name, inputs, output)

    for model in (_MODEL, _FALLBACK_MODEL):
        try:
            response = _get_client().models.generate_content(model=model, contents=prompt)
            return _parse_response(response.text, agent_name)
        except Exception as exc:
            last_exc = exc

    dims = DIMENSIONS[agent_name]
    return {
        dim: {"score": 1, "reasoning": f"Judge API call failed: {last_exc}"}
        for dim in dims
    }
