import google.generativeai as genai
import config

genai.configure(api_key=config.GEMINI_API_KEY)


def embed(text: str) -> list[float]:
    result = genai.embed_content(model="models/text-embedding-004", content=text)
    return result.embedding
