import json

from langchain_community.embeddings.ollama import OllamaEmbeddings

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

EMBEDDING_MODEL = config["EMBEDDING_MODEL"]


def get_embedding_function():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return embeddings
