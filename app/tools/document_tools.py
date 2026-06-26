from typing import List, Dict, Any
from langchain_core.documents import Document
from app.logger import get_logger

log = get_logger("tools.document_tools")

def summarize_corpus_tool(documents: List[Document]) -> Dict[str, Any]:
    """
    Summarizes the collection of uploaded documents.
    Computes file count, total characters, and unique sources.
    """
    log.info("Executing summarize_corpus tool...")
    if not documents:
        return {
            "file_count": 0,
            "total_characters": 0,
            "unique_sources": [],
            "summary_text": "No documents have been uploaded yet. The travel concierge corpus is currently empty."
        }
        
    unique_sources = list(set(doc.metadata.get("source", "Unknown Source") for doc in documents))
    total_chars = sum(len(doc.page_content) for doc in documents)
    
    summary_text = (
        f"Corpus Summary:\n"
        f"- Total unique files: {len(unique_sources)}\n"
        f"- Parsed chunks/pages: {len(documents)}\n"
        f"- Total text volume: {total_chars:,} characters\n"
        f"\nFiles uploaded:\n" + "\n".join(f"  • {src}" for src in unique_sources)
    )
    
    return {
        "file_count": len(unique_sources),
        "total_characters": total_chars,
        "unique_sources": unique_sources,
        "summary_text": summary_text
    }

def list_documents_tool(documents: List[Document]) -> str:
    """
    Returns a markdown-formatted list of all uploaded documents.
    """
    log.info("Executing list_documents tool...")
    if not documents:
        return "No documents have been uploaded yet."
        
    unique_sources = list(set(doc.metadata.get("source", "Unknown") for doc in documents))
    formatted_sources = sorted(unique_sources)
    
    return "### Ingested Travel Documents:\n" + "\n".join(f"- 📄 {src}" for src in formatted_sources)
