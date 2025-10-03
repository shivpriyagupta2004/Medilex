import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CORPUS_DIR = "corpus"
PERSIST_DIR = "chroma_db"

def ingest_documents():
    """
    Load documents from corpus directory and ingest into Chroma vector store.
    """
    print("🔄 Starting document ingestion...")
    
    # Check if corpus directory exists
    if not os.path.exists(CORPUS_DIR):
        print(f"❌ Corpus directory '{CORPUS_DIR}' not found!")
        print(f"Creating directory... Please add your text files there.")
        os.makedirs(CORPUS_DIR, exist_ok=True)
        return False
    
    # Load documents
    try:
        loader = DirectoryLoader(
            CORPUS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        
        if not documents:
            print(f"❌ No documents found in '{CORPUS_DIR}'")
            return False
        
        print(f"✅ Loaded {len(documents)} documents")
        
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return False
    
    # Split documents into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(splits)} chunks")
        
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return False
    
    # Create embeddings
    try:
        print("🔄 Creating embeddings (this may take a moment)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return False
    
    # Create vector store
    try:
        print(f"🔄 Creating vector store at '{PERSIST_DIR}'...")
        
        # Remove old database if exists
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
            print("🗑️  Removed old database")
        
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        
        print(f"✅ Vector store created successfully!")
        print(f"📊 Total vectors: {vectordb._collection.count()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return False

def main():
    """Main ingestion function."""
    success = ingest_documents()
    
    if success:
        print("\n✨ Ingestion complete! You can now run queries.")
    else:
        print("\n❌ Ingestion failed. Please check errors above.")

if __name__ == "__main__":
    main()