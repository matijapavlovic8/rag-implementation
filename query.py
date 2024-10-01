import argparse

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from config import CHROMA_PATH, MODEL, PROMPT_TEMPLATE_EN, OPENAI_API_BASE_URL, OPENAI_API_KEY
from embeddings import get_embedding_function
from reranker import rerank_results


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
    model = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=MODEL,
        base_url=OPENAI_API_BASE_URL
    )

    response_text = model.invoke(prompt).content
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text, sources

if __name__ == "__main__":
    main()
