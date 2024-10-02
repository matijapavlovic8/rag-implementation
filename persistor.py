import argparse
import os
import shutil
import pandas as pd
import nltk
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader

from config import DATA_PATH, CHROMA_PATH
from embeddings import get_embedding_function
from langchain_chroma import Chroma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    documents = []

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if pdf_files:
        print(f"Loading PDFs: {pdf_files}")
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents.extend(document_loader.load())

    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    for file_name in csv_files:
        print(f"Loading CSV: {file_name}")
        file_path = os.path.join(DATA_PATH, file_name)
        df = pd.read_csv(file_path, delimiter=";")

        for index, row in df.iterrows():
            text = ",".join(map(str, row.values))
            metadata = {"source": file_name, "row": index}
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def split_documents(documents: list[Document], chunk_size=800, chunk_overlap=80):
    chunks = []

    for doc in documents:
        paragraphs = doc.page_content.split('\n\n')
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            current_chunk = []
            current_length = 0
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_length + sentence_length <= chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    chunks.append(Document(
                        page_content=" ".join(current_chunk),
                        metadata=doc.metadata
                    ))
                    overlap_chunk = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_chunk + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
            if current_chunk:
                chunks.append(Document(
                    page_content=" ".join(current_chunk),
                    metadata=doc.metadata
                ))

    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page_or_row = chunk.metadata.get("page") or chunk.metadata.get("row")
        current_page_or_row_id = f"{source}:{page_or_row}"

        if current_page_or_row_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_or_row_id}:{current_chunk_index}"
        last_page_id = current_page_or_row_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
