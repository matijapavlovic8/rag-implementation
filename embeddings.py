from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL, OPENAI_API_BASE_URL, OPENAI_API_KEY, env


def get_embedding_function():
    if env == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY,
                                model=EMBEDDING_MODEL,
                                base_url=OPENAI_API_BASE_URL)
    else:
        return OllamaEmbeddings(model=EMBEDDING_MODEL)
