from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    State representing the data flow in the Travel Concierge RAG / Agent workflow.
    """
    messages: List[BaseMessage]          # Conversational message history
    query: str                           # Current user question/command
    documents: List[Document]            # Complete set of ingested documents (for tool operations)
    retrieved_context: str               # Raw text extracted from RAG similarity search
    source_documents: List[Document]     # Retrieved chunks (passed to UI for references)
    routing_decision: str                # Routed step: "direct", "rag", or "tool"
    tool_output: Optional[str]           # Resulting output from a triggered function/tool
    response: str                        # Final answer text
    metrics: Dict[str, Any]              # Timer stats & LangGraph tracing flags
