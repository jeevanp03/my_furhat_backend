"""
This module implements the DocumentAgent, which orchestrates a multi-step conversational workflow.
It integrates document retrieval via a RAG system, uncertainty checking, summarization, and response generation
using a chatbot. The workflow is modeled as a state graph with checkpointed memory to allow for resumption.

The module includes:
    - Global instances for RAG and chatbot initialization.
    - A custom State type (using TypedDict) to hold conversation state.
    - The DocumentAgent class with methods to capture input, retrieve document context, summarize content,
      check for uncertainty, generate clarification responses, and finally produce an answer via the chatbot.
"""

import os
import logging
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver

# Import custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import create_chatbot
from my_furhat_backend.utils.util import clean_output, summarize_text
# Import the RAG (Retrieval-Augmented Generation) class.
from my_furhat_backend.RAG.rag_flow import RAG

# Create a global instance of RAG with document ingestion parameters.
rag_instance = RAG(
    hf=True,
    persist_directory="my_furhat_backend/db",
    path_to_document="my_furhat_backend/ingestion/CMRPublished.pdf"
)

# Initialize the chatbot using a Llama model.
# Alternative: you can specify a custom model id by uncommenting and modifying the line below.
# chatbot = create_chatbot("llama", model_id="my_furhat_backend/ggufs_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
chatbot = create_chatbot("llama")

# Define the conversation state type using TypedDict.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    input: str

class DocumentAgent:
    """
    DocumentAgent orchestrates a multi-step conversational workflow.

    It utilizes a state graph to:
      - Capture user input.
      - Retrieve relevant document context using a RAG system.
      - Summarize the retrieved context.
      - Perform uncertainty checks on the retrieved context.
      - Provide a clarification prompt if uncertainty is detected.
      - Generate the final response using a chatbot.
      
    The agent uses persistent memory to checkpoint the conversation state and resume processing as needed.
    """
    def __init__(self):
        """
        Initialize the DocumentAgent.

        Sets up the persistent memory for checkpointing and builds the state graph defining the conversation flow.
        """
        # Initialize MemorySaver to checkpoint the state graph.
        self.memory = MemorySaver()
        # Create a state graph using the custom State schema.
        self.graph = StateGraph(State)
        # Build the conversation flow by adding nodes and edges.
        self._build_graph()
        # Compile the graph with checkpointing support.
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self):
        """
        Build the state graph for the conversational workflow.

        This method defines nodes for:
          - Input capture.
          - Document context retrieval.
          - Summarization.
          - Uncertainty checking.
          - Uncertainty response.
          - Answer generation.

        It also sets up edges for linear progression and conditional branching based on the uncertainty check.
        """
        # Add nodes to the graph with corresponding callback functions.
        self.graph.add_node("capture_input", self.input_node)
        self.graph.add_node("retrieval", self.retrieval_node)
        self.graph.add_node("uncertainty_check", self.uncertainty_check_node)
        self.graph.add_node("uncertainty_response", self.uncertainty_response_node)
        self.graph.add_node("generation", self.generation_node)
        self.graph.add_node("summarization", self.summarization_node)
    
        # Define a linear flow from the start node to the retrieval node.
        self.graph.add_edge(START, "capture_input")
        self.graph.add_edge("capture_input", "retrieval")
        self.graph.add_edge("retrieval", "summarization")
        self.graph.add_edge("summarization", "uncertainty_check")
    
        # Define conditional branching based on uncertainty.
        def condition_func(state):
            # Branch to uncertainty_response if uncertain; otherwise, proceed to generation.
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
        Capture user input and append it as a HumanMessage to the conversation state.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the 'messages' list containing the captured human message.
        """
        # Ensure that the 'messages' list exists in the state.
        state.setdefault("messages", [])
        # Create a HumanMessage using the user's input.
        human_msg = HumanMessage(content=state.get("input", ""))
        # Append the human message to the conversation history.
        state["messages"].append(human_msg)
        return {"messages": state["messages"]}
    
    def retrieval_node(self, state: State) -> dict:
        """
        Retrieve relevant document context using the RAG system and append a ToolMessage with the results.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with a ToolMessage containing the retrieved document context.
        """
        # Use the user's input as the query for retrieval.
        query = state.get("input", "")
        # Retrieve similar documents with reranking enabled.
        retrieved_docs = rag_instance.retrieve_similar(query, rerank=True)
        # Concatenate the content of retrieved documents or provide a default message if none are found.
        if retrieved_docs:
            retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)
        else:
            retrieved_text = "No relevant document context found."
        # Create a ToolMessage with a unique ID to store the retrieval results.
        tool_msg = ToolMessage(
            content=f"Retrieved context:\n{retrieved_text}",
            name="document_retriever",
            tool_call_id=str(uuid.uuid4())
        )
        # Append the retrieval message to the state's messages.
        state["messages"].append(tool_msg)
        return {"messages": state["messages"]}
    
    def summarization_node(self, state: State) -> dict:
        """
        Summarize the retrieved document context using a pre-trained summarization model.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with a new ToolMessage containing the summarized context.
        """
        # Locate the ToolMessage that holds the retrieved document context.
        retrieval_msg = next(
            (msg for msg in state["messages"]
             if isinstance(msg, ToolMessage) and msg.name == "document_retriever"),
            None
        )
        
        if retrieval_msg:
            # Remove any header text from the retrieval message.
            text_to_summarize = retrieval_msg.content.replace("Retrieved context:\n", "")
            # Generate a summary of the retrieved content.
            summarized_text = summarize_text(text_to_summarize)
        else:
            summarized_text = "No document context available to summarize."
        
        # Create a new ToolMessage for the summarized context.
        summary_msg = ToolMessage(
            content=f"Summarized context:\n{summarized_text}",
            name="summarizer",
            tool_call_id=str(uuid.uuid4())
        )
        # Append the summary message to the state's messages.
        state["messages"].append(summary_msg)
        return {"messages": state["messages"]}

    def uncertainty_check_node(self, state: State) -> dict:
        """
        Check for uncertainty in the retrieved document context.

        If the retrieved content is missing or too short, sets an 'uncertainty' flag in the state.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the 'uncertainty' flag added.
        """
        # Find the first ToolMessage in the state's messages.
        tool_msg = next((msg for msg in state["messages"] if isinstance(msg, ToolMessage)), None)
        # Set uncertainty to True if no retrieval message exists or its content is insufficient.
        state["uncertainty"] = (tool_msg is None) or (len(tool_msg.content) < 50)
        return {"messages": state["messages"], "uncertainty": state["uncertainty"]}
    
    def uncertainty_response_node(self, state: State) -> dict:
        """
        Provide a response to handle uncertainty in the retrieved context.

        Appends an AIMessage prompting the user to clarify their query if the document context is insufficient.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the uncertainty response message appended.
        """
        # Create an AIMessage asking the user for clarification.
        uncertain_msg = AIMessage(content="I'm not certain I have enough context from the document. Could you please clarify your question?")
        state["messages"].append(uncertain_msg)
        return {"messages": state["messages"]}
    
    def generation_node(self, state: State) -> dict:
        """
        Generate the final answer using the chatbot.

        Processes the current conversation state using the pre-initialized chatbot to produce an AI-generated response.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: The state updated with the AI-generated answer.
        """
        # Delegate response generation to the chatbot's conversation method.
        return chatbot.chatbot(state)
    
    def run(self, initial_input: str, system_prompt: str = None) -> str:
        """
        Execute the document agent's conversation flow.

        Initializes the conversation state with the user's input and an optional system prompt,
        processes the state through the compiled state graph, and returns the final AI-generated response.

        Parameters:
            initial_input (str): The user's query to initiate the conversation.
            system_prompt (str, optional): An optional system prompt to set the conversational context.
                If not provided, a default friendly prompt is used.

        Returns:
            str: The cleaned output from the final AI-generated message, or a default message if no response is produced.
        """
        # Use a default system prompt if none is provided.
        if system_prompt is None:
            system_prompt = (
                "You are a friendly and knowledgeable assistant who always communicates in a natural, conversational tone—like chatting with a friend. "
                "Use simple, clear language and a warm, approachable style. "
                "Rely solely on the content from the provided documents to craft your responses. "
                "Keep the conversation going like a human would by asking questions and providing helpful information. "
                "Engage the user by explaining the document content thoroughly and asking follow-up questions to clarify their needs. "
                "If the context is unclear, politely ask the user for more details. "
                "If you cannot answer based on the available document content, let the user know and invite them to rephrase or provide additional information. "
                "Keep the conversation engaging by prompting further discussion whenever appropriate."
            )
        # Initialize the conversation state with the system prompt as the first human message.
        state: State = {
            "input": initial_input,
            "messages": [HumanMessage(content=system_prompt)]
        }
        # Configuration settings for state graph execution (e.g., specifying a thread ID).
        config = {"configurable": {"thread_id": "1"}}
        # Process the conversation state through the compiled graph in streaming mode.
        for step in self.compiled_graph.stream(state, config, stream_mode="values"):
            pass  # The loop iterates through all steps without additional processing.
        # Retrieve the final AI message by scanning the state's messages in reverse order.
        final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
        # Clean the final output and return it; if no AI message is found, return a default message.
        return clean_output(final_ai_msg.content) if final_ai_msg else "No response generated."
