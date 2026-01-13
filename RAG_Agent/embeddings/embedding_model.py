"""Local embedding model using sentence-transformers."""

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """Returns a local embedding model using sentence-transformers."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    
    return embeddings

