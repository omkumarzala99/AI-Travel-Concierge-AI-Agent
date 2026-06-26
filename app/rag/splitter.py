from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.logger import get_logger, log_execution_time

log = get_logger("rag.splitter")

@log_execution_time("rag.splitter")
def split_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.
    """
    if not documents:
        log.info("No documents provided for chunking.")
        return []
        
    log.info(f"Chunking {len(documents)} document pages/sections with size={chunk_size}, overlap={chunk_overlap}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    log.info(f"Generated {len(chunks)} chunks from source documents.")
    return chunks
