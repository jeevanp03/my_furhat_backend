import os
import logging
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Import your custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import Chatbot_HuggingFace, Chatbot_LlamaCpp, create_chatbot
from my_furhat_backend.utils.util import clean_output

# Import your RAG class.
from my_furhat_backend.RAG.rag_flow import RAG

# Create a global instance of RAG.
rag_instance = RAG(
    hf=True,
    persist_directory="my_furhat_backend/db",
    path_to_document="my_furhat_backend/ingestion/CMRPublished.pdf"
)

# --- Define Chatbot Global Variable ---
# chatbot = create_chatbot("huggingface")
chatbot = create_chatbot("llama", model_id = "my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")

# --- Define State ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    input: str

# --- Graph Node Functions ---

# 1. Input Node: Capture the user's query and add it to the chat history.
def input_node(state: State) -> dict:
    state.setdefault("messages", [])
    human_msg = HumanMessage(content=state.get("input", ""))
    state["messages"].append(human_msg)
    return {"messages": state["messages"]}


# 2. Retrieval Node: Use the RAG flow to retrieve relevant document chunks.
def retrieval_node(state: State) -> dict:
    query = state.get("input", "")
    retrieved_docs = rag_instance.retrieve_similar(query, rerank=True)
    if retrieved_docs:
        retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)
    else:
        retrieved_text = "No relevant document context found."
    tool_msg = ToolMessage(
        content=f"Retrieved context:\n{retrieved_text}",
        name="document_retriever",
        tool_call_id=str(uuid.uuid4())
    )
    state["messages"].append(tool_msg)
    return {"messages": state["messages"]}

# 3. Uncertainty Check Node: Check if the retrieval returned enough context.
def uncertainty_check_node(state: State) -> dict:
    tool_msg = next((msg for msg in state["messages"] if isinstance(msg, ToolMessage)), None)
    # If the retrieved text is too short (less than 50 characters), mark as uncertain.
    state["uncertainty"] = (tool_msg is None) or (len(tool_msg.content) < 50)
    return {"messages": state["messages"], "uncertainty": state["uncertainty"]}

# 4a. Uncertainty Response Node: Ask for clarification if context is insufficient.
def uncertainty_response_node(state: State) -> dict:
    uncertain_msg = AIMessage(content="I'm not certain I have enough context from the document. Could you please clarify your question?")
    state["messages"].append(uncertain_msg)
    return {"messages": state["messages"]}

# 4b. Generation Node: Use the full conversation history to generate an answer based on the ingested document.
def generation_node(state: State) -> dict:
    # Use your custom chatbot. Here we choose the HuggingFace-based one.
    # The chatbot.chatbot() method formats the chat history, queries your LLM, cleans the response, and appends it.
    return chatbot.chatbot(state)

# --- Build the Graph Workflow ---
graph = StateGraph(State)
graph.add_node("capture_input", input_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("uncertainty_check", uncertainty_check_node)
graph.add_node("uncertainty_response", uncertainty_response_node)
graph.add_node("generation", generation_node)

graph.add_edge(START, "capture_input")
graph.add_edge("capture_input", "retrieval")
graph.add_edge("retrieval", "uncertainty_check")

# Conditional branch: if uncertainty flag is set, route to uncertainty_response; otherwise, to generation.
def condition_func(state):
    return "uncertainty_response" if state.get("uncertainty") else "generation"

graph.add_conditional_edges(
    "uncertainty_check",
    condition_func,
    {"uncertainty_response": "uncertainty_response", "generation": "generation"}
)
graph.add_edge("uncertainty_response", END)
graph.add_edge("generation", END)

memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)

# --- Run the Conversation ---
def run_conversation_streaming():
    config = {"configurable": {"thread_id": "1"}}
    print("Welcome. This assistant will discuss only the ingested document.")
    query = input("Enter your query about the document (or 'exit' to quit): ")
    if query.strip().lower() == "exit":
        return
    
    # Set up initial state with a system prompt guiding the conversation.
    system_prompt = SystemMessage(content=(
        "You are a friendly and knowledgeable assistant. "
        "When answering, speak naturally and conversationally—as if you're chatting with a friend. "
        "Use simple, clear language and explain things in a warm, approachable manner. "
        "Answer only based on the provided document content."
    ))

    state: State = {
        "input": query,
        "messages": [HumanMessage(content=system_prompt.content)]
    }
    
    print("Streaming response...")
    for step in compiled_graph.stream(state, config, stream_mode="values"):
        pass  # Each step updates state; final state holds the complete history.
    
    # Instead of printing the entire conversation, print only the final AI response.
    final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
    if final_ai_msg:
        print("Assistant:", clean_output(final_ai_msg.content))
    else:
        print("No AI response found.")

    # Continue conversation loop.
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        state["input"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        for step in compiled_graph.stream(state, config, stream_mode="values"):
            pass
        final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
        if final_ai_msg:
            print("Assistant:", clean_output(final_ai_msg.content))
        else:
            print("No AI response found.")


if __name__ == "__main__":
    run_conversation_streaming()
