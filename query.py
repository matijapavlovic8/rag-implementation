import argparse
import json

import openai
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from embeddings import get_embedding_function

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

CHROMA_PATH = config["CHROMA_PATH"]
PROMPT_TEMPLATE_EN = config["PROMPT_TEMPLATE"]["en"]
PROMPT_TEMPLATE_REL = config["PROMPT_TEMPLATE"]["relevance"]
MODEL = config["MODEL"]
client = openai.OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def rerank_results(results, query_text):
    reranked_results = []

    for doc, score in results:
        relevance_prompt = PROMPT_TEMPLATE_REL.format(document=doc.page_content, query=query_text)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": relevance_prompt}
            ]
        )

        relevance_score = float(response.choices[0].message.content)
        reranked_results.append((doc, relevance_score))

    reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
    print(reranked_results)

    return reranked_results


if __name__ == "__main__":
    main()
