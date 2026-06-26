from typing import List
from langchain_core.documents import Document
from app.rag.vectorstore import load_vector_store
from app.config import RETRIEVAL_K
from app.logger import get_logger, log_execution_time

log = get_logger("rag.retriever")

@log_execution_time("rag.retriever")
def retrieve_relevant_documents(query: str, vector_store = None, k: int = RETRIEVAL_K) -> List[Document]:
    """
    Retrieves top-k relevant documents from the vector store for a given query.
    If no vector_store is passed, attempts to load the default local FAISS store.
    """
    vs = vector_store or load_vector_store()
    if vs is None:
        log.warning("Vector store is not initialized or could not be loaded. Returning empty retrieval list.")
        return []
        
    log.info(f"Running similarity search (k={k}) for query: '{query}'")
    try:
        results = vs.similarity_search(query, k=k)
        log.info(f"Successfully retrieved {len(results)} chunks.")
        return results
    except Exception as e:
        log.error(f"Error executing similarity search: {str(e)}", exc_info=True)
        return []
