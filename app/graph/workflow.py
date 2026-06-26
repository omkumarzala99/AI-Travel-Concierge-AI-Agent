from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.graph.nodes import classifier_node, retriever_node, tool_node, generator_node
from app.logger import get_logger

log = get_logger("graph.workflow")

def route_after_classifier(state: AgentState) -> str:
    """
    Conditional edge router determining where to go from classifier_node.
    """
    decision = state.get("routing_decision", "direct")
    log.info(f"Routing edge evaluating classifier decision: '{decision}'")
    
    if decision == "tool":
        return "tool"
    elif decision == "rag":
        return "retriever"
    else:
        return "generator"

def build_workflow() -> StateGraph:
    """
    Compiles and returns the LangGraph workflow StateGraph.
    """
    workflow = StateGraph(AgentState)
    
    # Register all nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("generator", generator_node)
    
    # Establish entry point
    workflow.set_entry_point("classifier")
    
    # Setup conditional edge from classifier node
    workflow.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "tool": "tool",
            "retriever": "retriever",
            "generator": "generator"
        }
    )
    
    # Setup standard edges
    workflow.add_edge("retriever", "generator")
    
    # Setup exit points
    workflow.add_edge("generator", END)
    workflow.add_edge("tool", END)
    
    # Compile the graph
    app = workflow.compile()
    log.info("LangGraph Travel Concierge workflow successfully compiled.")
    return app

# Expose compiled app instance
graph_app = build_workflow()
