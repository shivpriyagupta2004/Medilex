import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

def run_query(question: str, top_k: int = 3, verbose: bool = False) -> str:
    """
    Query the vector store and return relevant information.
    
    Args:
        question: The query string
        top_k: Number of documents to retrieve
        verbose: Whether to print debug information
        
    Returns:
        Formatted answer string
    """
    if not question or not isinstance(question, str):
        return "❌ Please provide a valid question."
    
    # Check if vector store exists
    if not os.path.exists(PERSIST_DIR):
        return (
            "❌ Knowledge base not found!\n"
            "Please run 'python rag/ingest.py' first to create the database."
        )
    
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        
        # Create retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return "❌ No relevant information found in the knowledge base."
        
        if verbose:
            print(f"📚 Retrieved {len(docs)} documents")
        
        # Format response
        answer = "✅ **Here's what I found:**\n\n"
        
        for i, doc in enumerate(docs, 1):
            snippet = doc.page_content.strip()
            # Clean up the snippet
            snippet = snippet.replace("\n\n", "\n").replace("\n", " ")
            
            # Truncate if too long
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            
            source = doc.metadata.get("source", "unknown.txt")
            source_name = os.path.basename(source)
            
            answer += f"**{i}. From {source_name}:**\n{snippet}\n\n"
        
        # Add a helpful summary
        answer += "\n💡 **Summary:**\n"
        answer += "The above information is from our medical knowledge base. "
        answer += "Always consult with a healthcare professional for personalized advice.\n"
        
        return answer
        
    except Exception as e:
        return f"❌ Error querying knowledge base: {str(e)}"

def search_medicine(medicine_name: str) -> str:
    """
    Search for specific medicine information.
    
    Args:
        medicine_name: Name of the medicine
        
    Returns:
        Information about the medicine
    """
    query = f"What is {medicine_name}? Uses, dosage, and side effects of {medicine_name}"
    return run_query(query, top_k=2)

def search_symptom(symptom: str) -> str:
    """
    Search for symptom-related information.
    
    Args:
        symptom: Name of the symptom
        
    Returns:
        Information about the symptom
    """
    query = f"What causes {symptom}? Treatment and self-care for {symptom}"
    return run_query(query, top_k=3)

if __name__ == "__main__":
    # Test queries
    print("=== Testing RAG Query System ===\n")
    
    test_queries = [
        "How to treat cough?",
        "What is paracetamol used for?",
        "Self-care for fever"
    ]
    
    for q in test_queries:
        print(f"Query: {q}")
        print(run_query(q, top_k=2, verbose=True))
        print("\n" + "="*60 + "\n")