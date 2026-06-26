import os
from dotenv import load_dotenv

# Load standard environment variables from .env file
load_dotenv()

# Retrieve API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Auto-detect default provider based on available keys
def detect_provider():
    if GOOGLE_API_KEY:
        return "gemini"
    elif GROQ_API_KEY:
        return "groq"
    elif OPENAI_API_KEY:
        return "openai"
    return "gemini"  # Default fallback if no keys are found

LLM_PROVIDER = os.getenv("LLM_PROVIDER", detect_provider()).lower()

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-1.5-flash",
    "groq": "llama3-8b-8192",
    "openai": "gpt-4o-mini"
}

LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(LLM_PROVIDER, "gemini-1.5-flash"))

# Embedding Provider: "gemini", "openai", "local"
# Auto-detect default: use gemini if Gemini API key is present, otherwise fallback to local
def detect_embedding_provider():
    if GOOGLE_API_KEY:
        return "gemini"
    return "local"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", detect_embedding_provider()).lower()

# LangSmith / Observability
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-travel-concierge-rag")

# Vector Store configurations
# Saves index inside local project folder
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faiss_db")

# Chunk configurations
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
