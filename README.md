# RAG implementation

This project comprises a ChromaDB instance filled with various files in a PDF format and a RAG used for effective and precise search of said documents.
Result of the search is LLM generated output alongside an exact position in a file from which the output was generated.

As an example the database was filled with some popular board games rules PDFs.

## How to run?
To run this project locally Ollama (https://ollama.com/) needs to be installed. Additionally, embedding and inference models have to be pulled.
I used llama3 and nomic. Alternatively RAG can employ various non-local models (OpenAI, Amazon Bedrock...)

