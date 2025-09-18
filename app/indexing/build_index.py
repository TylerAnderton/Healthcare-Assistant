import argparse
import os
import glob
import pandas as pd
from typing import List
import logging

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# logger = logging.getLogger(__name__)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_corpus(corpus_dir: str) -> List[Document]:
    docs: List[Document] = []
    for fp in glob.glob(os.path.join(corpus_dir, "*.parquet")):
        print(f"Loading corpus from {fp}")
        df = pd.read_parquet(fp)
        print(f"Attempting to load {len(df)} rows")
        n_rows = 0
        for _, row in df.iterrows():
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            metadata = {k: row[k] for k in row.index if k != "text"}
            docs.append(Document(page_content=text, metadata=metadata))
            n_rows += 1
        print(f"Loaded {n_rows} rows from {fp}")
    print(f"Loaded {len(docs)} documents from corpus")
    return docs


def build_index(corpus_dir: str, store_dir: str, embedding_model: str):
    ensure_dir(store_dir)
    docs = load_corpus(corpus_dir)
    if not docs:
        print("No corpus documents found; aborting index build.")
        return
    # Split into smaller chunks for better recall
    print(f"Splitting {len(docs)} documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = splitter.split_documents(docs)
    print(f"Embedding {len(docs)} document chunks")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=store_dir)
    print(f"Built index with {len(docs)} document chunks at {store_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.getenv("PROCESSED_DIR", "./data/processed") + "/corpus")
    parser.add_argument("--store", default=os.getenv("VECTORSTORE_DIR", "./data/processed/vectorstore"))
    parser.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    args = parser.parse_args()

    build_index(args.corpus, args.store, args.embedding_model)
