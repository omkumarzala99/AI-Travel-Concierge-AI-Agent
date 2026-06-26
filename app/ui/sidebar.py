import streamlit as st
import os
from app.config import (
    LLM_PROVIDER, LLM_MODEL, EMBEDDING_PROVIDER,
    GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, MAX_FILE_SIZE_MB
)
from app.rag.loaders import load_document_from_bytes
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vector_store, save_vector_store, clear_vector_store
from app.utils.helpers import allowed_file, validate_file_size, format_file_size
from app.logger import get_logger

log = get_logger("ui.sidebar")

def render_sidebar():
    """
    Renders the sidebar controls, handles document uploading, configuration overrides,
    and runs the RAG ingestion pipeline.
    """
    with st.sidebar:
        # Title & Logo
        st.markdown("<h2 style='text-align: center;'>✈️ Concierge Panel</h2>", unsafe_allow_html=True)
        st.markdown("---")


        # 2. Document Upload Section
        st.subheader("📄 Upload Travel Documents")
        st.caption("Upload flight tickets, hotel reservations, or travel itineraries (PDF, TXT, DOCX).")
        
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Ingestion logic when files are modified
        if uploaded_files:
            new_files_detected = False
            for f in uploaded_files:
                if f.name not in st.session_state["processed_files"]:
                    new_files_detected = True
                    break
                    
            if new_files_detected:
                with st.spinner("Processing uploaded documents..."):
                    all_new_documents = []
                    
                    for uploaded_file in uploaded_files:
                        file_name = uploaded_file.name
                        file_bytes = uploaded_file.read()
                        
                        # Validate extension and size
                        if not allowed_file(file_name):
                            st.error(f"Unsupported format: {file_name}")
                            continue
                        if not validate_file_size(len(file_bytes), MAX_FILE_SIZE_MB):
                            st.error(f"File too large (> {MAX_FILE_SIZE_MB}MB): {file_name}")
                            continue
                            
                        try:
                            # Load and parse
                            docs = load_document_from_bytes(file_bytes, file_name)
                            all_new_documents.extend(docs)
                            st.session_state["processed_files"].add(file_name)
                            log.info(f"Loaded {len(docs)} document elements from '{file_name}'")
                        except Exception as e:
                            st.error(f"Error parsing '{file_name}': {str(e)}")
                            
                    if all_new_documents:
                        # Append new document objects to session state
                        st.session_state["documents"].extend(all_new_documents)
                        
                        # Split all documents into chunks
                        chunks = split_documents(st.session_state["documents"])
                        st.session_state["chunks"] = chunks
                        
                        try:
                            # Recompile vector store index
                            vector_store = create_vector_store(chunks)
                            st.session_state["vector_store"] = vector_store
                            save_vector_store(vector_store)
                            st.success(f"Successfully indexed {len(uploaded_files)} files!")
                        except Exception as e:
                            st.error(f"Failed to create vector store: {str(e)}")
                            log.error(f"Vector store compilation failure: {str(e)}", exc_info=True)
                            
        # 3. Display Corpus Status and Info
        st.markdown("---")
        st.subheader("📊 Ingested Corpus Status")
        
        if st.session_state["processed_files"]:
            st.info(f"Indexed Files: {len(st.session_state['processed_files'])}")
            for file_name in sorted(st.session_state["processed_files"]):
                st.markdown(f"- 📄 {file_name}")
            st.markdown(f"**Total Parsed Chunks:** {len(st.session_state.get('chunks', []))}")
        else:
            st.warning("No travel files indexed yet.")
            
        # 4. Clear state & memory button
        st.markdown("---")
        if st.button("🗑️ Clear Database & Chat", use_container_width=True):
            # Reset session state vars
            st.session_state["messages"] = []
            st.session_state["documents"] = []
            st.session_state["chunks"] = []
            st.session_state["processed_files"] = set()
            st.session_state["vector_store"] = None
            
            # Clear local FAISS folder
            clear_vector_store()
            
            st.rerun()
            
        # Footer
        st.markdown("<p style='font-size: 0.8rem; text-align: center; color: gray;'>AI Travel Concierge Agent v1.0.0 (Weeks 1-2)</p>", unsafe_allow_html=True)
