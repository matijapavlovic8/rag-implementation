from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import MODEL, PROMPT_TEMPLATE_REL, OPENAI_API_BASE_URL, OPENAI_API_KEY


def rerank_results(results, query_text):
    model = ChatOpenAI(
        base_url=OPENAI_API_BASE_URL,
        model=MODEL,
        api_key=OPENAI_API_KEY
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_REL)
    scored_results = []

    for doc, _ in results:
        context_text = doc.page_content
        prompt = prompt_template.format(context=context_text, query=query_text)

        response = model.invoke(prompt)
        score = float(response.content.strip())
        scored_results.append((doc, score))

    ranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

    return ranked_results
