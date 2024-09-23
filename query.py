import argparse
import json

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embeddings import get_embedding_function

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

CHROMA_PATH = config["CHROMA_PATH"]
PROMPT_TEMPLATE_EN = config["PROMPT_TEMPLATE"]["en"]
PROMPT_TEMPLATE_REL = config["PROMPT_TEMPLATE"]["relevance"]
MODEL = config["MODEL"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EN)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=MODEL)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def rerank_results(results, query_text):
    model = Ollama(model=MODEL)
    reranked_results = []

    for doc, score in results:
        relevance_prompt = PROMPT_TEMPLATE_REL.format(document=doc.page_content, query=query_text)
        relevance_score = float(model.invoke(relevance_prompt))
        reranked_results.append((doc, relevance_score))

    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
    print(reranked_results)

    return reranked_results


if __name__ == "__main__":
    main()
