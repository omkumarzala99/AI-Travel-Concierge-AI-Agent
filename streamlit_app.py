import streamlit as st
from app.ui.sidebar import render_sidebar
from app.ui.chat import render_chat_interface
from app.rag.vectorstore import load_vector_store

# 1. Page Configuration
st.set_page_config(
    page_title="AI Travel Concierge AI Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Session State Initialization
if "documents" not in st.session_state:
    st.session_state["documents"] = []

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

if "vector_store" not in st.session_state:
    # Attempt to restore FAISS vector db from local cache if present
    st.session_state["vector_store"] = load_vector_store()

# 3. Render Components
render_sidebar()
render_chat_interface()
