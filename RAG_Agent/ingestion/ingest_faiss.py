"""FAISS knowledge base ingestion."""

import os
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RAG_Agent.embeddings.embedding_model import get_embedding_model


def ingest_knowledge_base():
    """Ingests knowledge base files into FAISS vector store."""
    project_root = Path(__file__).parent.parent.parent
    
    faiss_db_path = project_root / "faiss_db"
    if faiss_db_path.exists() and any(faiss_db_path.iterdir()):
        print("FAISS database already exists. Skipping ingestion.")
        return
    
    kb_dir = project_root / "knowledge_base"
    if not kb_dir.exists():
        print(f"Knowledge base directory not found: {kb_dir}")
        return
    
    txt_files = list(kb_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {kb_dir}")
        return
    
    print(f"Found {len(txt_files)} knowledge base files")
    
    print("Loading embedding model...")
    embeddings = get_embedding_model()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_documents = []
    
    for txt_file in txt_files:
        print(f"Processing {txt_file.name}...")
        
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = text_splitter.split_text(content)
        print(f"  Split into {len(chunks)} chunks")
        
        for chunk_id, chunk_text in enumerate(chunks, start=1):
            source = txt_file.stem
            # Normalize source name to match intent (limitation -> limitations)
            if source == "limitation":
                source = "limitations"
            file_path = str(txt_file.relative_to(project_root))
            
            doc = {
                "page_content": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                    "content_type": "knowledge_base"
                }
            }
            all_documents.append(doc)
    
    print(f"Total documents created: {len(all_documents)}")
    
    print("Creating FAISS vector store...")
    texts = [doc["page_content"] for doc in all_documents]
    metadatas = [doc["metadata"] for doc in all_documents]
    
    vectorstore = FAISS.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings
    )
    
    print(f"Saving FAISS database to {faiss_db_path}...")
    vectorstore.save_local(str(faiss_db_path))
    
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest_knowledge_base()

