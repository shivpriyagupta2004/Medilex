# rag/query.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

def run_query(question: str, top_k: int = 3) -> str:
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the persisted vector DB
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "❌ No relevant documents found in knowledge base."

    # Format answer
    answer = "✅ Here’s what I found:\n\n"
    for i, doc in enumerate(docs, 1):
        snippet = doc.page_content.strip().replace("\n", " ")
        source = doc.metadata.get("source", "unknown.txt")
        answer += f"{i}. ({source}) {snippet}\n\n"
    return answer
