import os
import shutil
import hashlib
from typing import List, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import (
    EMBEDDING_PROVIDER,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    VECTOR_STORE_DIR
)
from app.logger import get_logger, log_execution_time

log = get_logger("rag.vectorstore")


class SimpleHashingEmbeddings(Embeddings):
    """
    A lightweight, zero-dependency local embedding class that computes
    a normalized feature vector using token hashing. Runs instantly on CPU,
    requiring no heavy deep-learning frameworks (like PyTorch or sentence-transformers).
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        text = text.lower().strip()
        words = text.split()
        
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not words:
            return vector.tolist()
            
        for word in words:
            # MD5 hash to compute token feature mapping
            h = hashlib.md5(word.encode('utf-8')).hexdigest()
            idx = int(h, 16) % self.dimension
            vector[idx] += 1.0
            
        # Normalize vector using L2 Norm
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings():
    """
    Returns the configured embeddings instance based on EMBEDDING_PROVIDER config.
    Falls back to a zero-dependency Hashing embedding if local HuggingFace libraries are missing.
    """
    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set but OpenAI embeddings provider was requested.")
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
    elif EMBEDDING_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set but Gemini embeddings provider was requested.")
        # If Gemini embeddings fail due to account v1beta API permissions, they can override to "local"
        try:
            return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
        except Exception as e:
            log.warning(f"Failed to load Gemini embeddings API: {str(e)}. Falling back to local Hashing embeddings...")
            return SimpleHashingEmbeddings()
        
    elif EMBEDDING_PROVIDER == "local":
        log.info("Initializing local embeddings...")
        try:
            # Try to load HuggingFace in case package is present
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            log.warning(
                f"Could not load HuggingFaceEmbeddings: {str(e)}. "
                "Falling back to zero-dependency local SimpleHashingEmbeddings..."
            )
            return SimpleHashingEmbeddings()
            
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")


@log_execution_time("rag.vectorstore")
def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Creates a new FAISS vector store from a list of Document chunks.
    """
    embeddings = get_embeddings()
    log.info(f"Creating FAISS vector store with {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def save_vector_store(vector_store: FAISS, path: str = VECTOR_STORE_DIR):
    """
    Saves the FAISS index to the local filesystem.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vector_store.save_local(path)
        log.info(f"Successfully saved FAISS index locally to: {path}")
    except Exception as e:
        log.error(f"Error saving FAISS index to {path}: {str(e)}", exc_info=True)
        raise

def load_vector_store(path: str = VECTOR_STORE_DIR) -> Optional[FAISS]:
    """
    Loads the FAISS index from the local filesystem.
    """
    if not os.path.exists(path) or not os.path.exists(os.path.join(path, "index.faiss")):
        log.warning(f"No FAISS index found at {path}")
        return None
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        log.info(f"Successfully loaded FAISS index from {path}")
        return vector_store
    except Exception as e:
        log.error(f"Error loading FAISS index from {path}: {str(e)}", exc_info=True)
        return None

def clear_vector_store(path: str = VECTOR_STORE_DIR):
    """Clears the local FAISS index folder if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path)
        log.info(f"Cleared FAISS vector store directory: {path}")
