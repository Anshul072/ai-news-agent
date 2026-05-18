from google import genai
import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)


def embed(text: str) -> list[float]:
    result = _client.models.embed_content(model="text-embedding-004", contents=text)
    return list(result.embeddings[0].values)
