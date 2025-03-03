import os
import logging
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver

# Import your custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import create_chatbot
from my_furhat_backend.utils.util import clean_output
# Import your RAG class.
from my_furhat_backend.RAG.rag_flow import RAG

# Create a global instance of RAG with document ingestion parameters.
rag_instance = RAG(
    hf=True,
    persist_directory="my_furhat_backend/db",
    path_to_document="my_furhat_backend/ingestion/CMRPublished.pdf"
)

# Initialize your chatbot using a Llama model.
# chatbot = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")

chatbot = create_chatbot("llama")

# Define the conversation state type using TypedDict.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    input: str

class DocumentAgent:
    """
    DocumentAgent orchestrates a multi-step conversational workflow.
    
    It uses a state graph to process input, retrieve context from documents using a RAG system,
    perform uncertainty checks, and finally generate a response using a chatbot. The agent
    leverages persistent memory to save and resume conversation state.
    """
    def __init__(self):
        """
        Initialize the DocumentAgent.

        Sets up memory for checkpointing and builds the state graph for conversation flow.
        """
        # MemorySaver to checkpoint the graph state.
        self.memory = MemorySaver()
        # Create a state graph with the defined State schema.
        self.graph = StateGraph(State)
        self._build_graph()
        # Compile the graph, using memory for checkpoints.
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self):
        """
        Build the state graph for the conversational workflow.
        
        Defines nodes corresponding to input capture, retrieval of document context, uncertainty checking,
        uncertainty response, and answer generation. Also sets up the edges and conditional branching
        in the graph.
        """
        # Define nodes in the conversation graph.
        self.graph.add_node("capture_input", self.input_node)
        self.graph.add_node("retrieval", self.retrieval_node)
        self.graph.add_node("uncertainty_check", self.uncertainty_check_node)
        self.graph.add_node("uncertainty_response", self.uncertainty_response_node)
        self.graph.add_node("generation", self.generation_node)
    
        # Define the linear flow from start to retrieval.
        self.graph.add_edge(START, "capture_input")
        self.graph.add_edge("capture_input", "retrieval")
        self.graph.add_edge("retrieval", "uncertainty_check")
    
        # Define a conditional branch based on uncertainty check.
        def condition_func(state):
            # If the state is uncertain, branch to uncertainty_response; otherwise, to generation.
            return "uncertainty_response" if state.get("uncertainty") else "generation"
    
        self.graph.add_conditional_edges(
            "uncertainty_check",
            condition_func,
            {"uncertainty_response": "uncertainty_response", "generation": "generation"}
        )
        # Both branches eventually lead to the end of the conversation.
        self.graph.add_edge("uncertainty_response", END)
        self.graph.add_edge("generation", END)
    
    def input_node(self, state: State) -> dict:
        """
        Capture user input and append it as a HumanMessage.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state containing the messages list with the captured human message.
        """
        # Ensure the messages list is initialized.
        state.setdefault("messages", [])
        # Create a HumanMessage from the 'input' field.
        human_msg = HumanMessage(content=state.get("input", ""))
        state["messages"].append(human_msg)
        return {"messages": state["messages"]}
    
    def retrieval_node(self, state: State) -> dict:
        """
        Retrieve relevant document context using RAG and append a ToolMessage.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with a ToolMessage containing retrieved document context.
        """
        # Use the user input as the query.
        query = state.get("input", "")
        # Retrieve similar documents using the global RAG instance with reranking.
        retrieved_docs = rag_instance.retrieve_similar(query, rerank=True)
        # Concatenate retrieved document content or set a default message if none found.
        if retrieved_docs:
            retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)
        else:
            retrieved_text = "No relevant document context found."
        # Create a ToolMessage to capture the retrieval result with a unique tool call ID.
        tool_msg = ToolMessage(
            content=f"Retrieved context:\n{retrieved_text}",
            name="document_retriever",
            tool_call_id=str(uuid.uuid4())
        )
        state["messages"].append(tool_msg)
        return {"messages": state["messages"]}
    
    def uncertainty_check_node(self, state: State) -> dict:
        """
        Check for uncertainty in the retrieved context.

        Determines if the retrieved document context is insufficient (e.g., too short).

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state including an 'uncertainty' flag.
        """
        # Retrieve the first ToolMessage from the messages.
        tool_msg = next((msg for msg in state["messages"] if isinstance(msg, ToolMessage)), None)
        # Set uncertainty to True if no tool message is found or if the content is too short.
        state["uncertainty"] = (tool_msg is None) or (len(tool_msg.content) < 50)
        return {"messages": state["messages"], "uncertainty": state["uncertainty"]}
    
    def uncertainty_response_node(self, state: State) -> dict:
        """
        Provide a response when uncertainty is detected.

        Appends an AIMessage asking for clarification if the retrieved context is insufficient.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the uncertainty response appended.
        """
        # Create an AIMessage to ask for clarification due to uncertainty.
        uncertain_msg = AIMessage(content="I'm not certain I have enough context from the document. Could you please clarify your question?")
        state["messages"].append(uncertain_msg)
        return {"messages": state["messages"]}
    
    def generation_node(self, state: State) -> dict:
        """
        Generate the final answer using the chatbot.

        Uses the pre-initialized chatbot to process the current state and generate a response.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: The state updated with the AI-generated response.
        """
        # Invoke the chatbot's conversational method to generate an answer.
        return chatbot.chatbot(state)
    
    def run(self, initial_input: str, system_prompt: str = None) -> str:
        """
        Execute the document agent's conversation flow.

        Initializes the conversation state with the initial input and an optional system prompt,
        processes the conversation through the state graph, and returns the final AI message.

        Parameters:
            initial_input (str): The user's initial query.
            system_prompt (str, optional): An optional system prompt to set the conversational context.
                Defaults to a friendly and informative prompt if not provided.

        Returns:
            str: The cleaned output from the final AI-generated message, or a default message if no response is generated.
        """
        # Set a default system prompt if none is provided.
        if system_prompt is None:
            system_prompt = (
                "You are a friendly and knowledgeable assistant. "
                "When answering, speak naturally and conversationally—as if you're chatting with a friend. "
                "Use simple, clear language and explain things in a warm, approachable manner. "
                "Answer only based on the provided document content."
            )
        # Initialize the conversation state with the system prompt as a starting HumanMessage.
        state: State = {
            "input": initial_input,
            "messages": [HumanMessage(content=system_prompt)]
        }
        # Configuration for the state graph execution, here including a thread ID.
        config = {"configurable": {"thread_id": "1"}}
        # Process the conversation state through the compiled graph.
        for step in self.compiled_graph.stream(state, config, stream_mode="values"):
            pass
        # Retrieve the final AI message from the state messages.
        final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
        # Clean and return the output, or a default message if no AI response is found.
        return clean_output(final_ai_msg.content) if final_ai_msg else "No response generated."
