"""Context retrieval tool using FAISS vector search."""

import os
import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RAG_Agent.embeddings.embedding_model import get_embedding_model


_vectorstore = None


def _get_vectorstore():
    """Loads and caches the FAISS vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        project_root = Path(__file__).parent.parent.parent
        faiss_db_path = project_root / "faiss_db"
        if not faiss_db_path.exists():
            raise FileNotFoundError("FAISS database not found. Run ingestion first.")
        
        embeddings = get_embedding_model()
        _vectorstore = FAISS.load_local(
            str(faiss_db_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore


@tool
def retrieve_context(query: str, intents: list[str]) -> str:
    """Retrieves relevant context from FAISS based on detected intents."""
    non_business_intents = [intent for intent in intents if intent != "business"]
    if not non_business_intents:
        return ""
    
    try:
        vectorstore = _get_vectorstore()
    except FileNotFoundError as e:
        print(f"Error loading vectorstore: {e}")
        return ""
    
    all_contexts = []
    
    for intent in non_business_intents:
        results = vectorstore.similarity_search_with_score(
            query,
            k=10
        )
        
        filtered_results = []
        for doc, score in results:
            if doc.metadata.get("source") == intent:
                filtered_results.append(doc)
                if len(filtered_results) >= 3:
                    break
        
        for doc in filtered_results:
            all_contexts.append(doc.page_content)
    
    combined_context = "\n\n".join(all_contexts)
    
    return combined_context

