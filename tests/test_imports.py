def test_imports():
    """
    Smoke test to verify that all primary external packages and project 
    modules can be imported cleanly without exceptions.
    """
    # Verify core external packages
    import streamlit
    import langchain
    import langgraph
    import pypdf
    import docx
    import dotenv
    
    # Verify internal project modules
    from app.config import LLM_PROVIDER, LLM_MODEL, EMBEDDING_PROVIDER
    from app.logger import logger
    from app.utils.helpers import allowed_file, validate_file_size
    from app.rag.loaders import load_document_from_bytes
    from app.rag.splitter import split_documents
    from app.rag.vectorstore import get_embeddings
    from app.rag.retriever import retrieve_relevant_documents
    from app.tools.document_tools import summarize_corpus_tool
    from app.graph.state import AgentState
    from app.graph.nodes import classifier_node, retriever_node, generator_node
    from app.graph.workflow import graph_app
    
    assert graph_app is not None
    assert LLM_PROVIDER in ["gemini", "groq", "openai"]
