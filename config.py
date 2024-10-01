import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")

env = os.getenv("ENV")

PROMPT_TEMPLATE_EN = os.getenv("PROMPT_TEMPLATE_EN")
PROMPT_TEMPLATE_REL = os.getenv("PROMPT_TEMPLATE_REL")


if env == "local":
    CHROMA_PATH = os.getenv("CHROMA_PATH_LOCAL")
    MODEL = os.getenv("MODEL_LOCAL")
    OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL_LOCAL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_LOCAL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_LOCAL")
elif env == "openai":
    CHROMA_PATH = os.getenv("CHROMA_PATH_OPENAI")
    MODEL = os.getenv("MODEL_OPENAI")
    OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL_OPENAI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_OPENAI")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_OPENAI")
else:
    raise ValueError("Invalid ENV specified in .env file.")

print(f"Using {env} configuration")
print(f"CHROMA_PATH: {CHROMA_PATH}")
print(f"Model: {MODEL}")
print(f"API Base URL: {OPENAI_API_BASE_URL}")
