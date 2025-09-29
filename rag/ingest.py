from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

CORPUS_DIR = "corpus"
PERSIST_DIR = "chroma_db"

def main():
    docs = []
    for file in os.listdir(CORPUS_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(CORPUS_DIR, file))
            docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(docs)

    # ✅ Offline embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"✅ Ingested {len(docs_split)} text chunks into {PERSIST_DIR}")

if __name__ == "__main__":
    main()
