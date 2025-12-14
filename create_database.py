from pathlib import Path
import os
import shutil

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = Path("chroma")
DATA_PATH = Path("data")

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(
        str(DATA_PATH),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True,
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No PDF files found in {DATA_PATH}")
    return docs

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} docs into {len(chunks)} chunks")

    if chunks:
        sample = chunks[min(10, len(chunks)-1)]
        print(sample.page_content[:300])  
        print(sample.metadata)
    return chunks

def save_to_chroma(chunks):
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH.resolve()}")

if __name__ == "__main__":
    main()
