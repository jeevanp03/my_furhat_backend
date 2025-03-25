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
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

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
llm = chatbot.llm

# Define the conversation state type using TypedDict.
class State(TypedDict):
    messages: Annotated[List[BaseMessage], "add_messages"]
    input: str

class QuestionCache:
    """
    A class to manage caching of questions and answers with similarity checking.
    """
    def __init__(self, cache_file: str = "question_cache.json"):
        self.cache_file = cache_file
        # Ensure the cache file exists
        self._ensure_cache_file()
        self.cache: Dict[str, Dict] = self._load_cache()
        # Initialize the sentence transformer model for similarity checking
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _normalize_question(self, question: str) -> str:
        """
        Normalize the question by removing common words and standardizing format.
        
        Parameters:
            question (str): The input question.
            
        Returns:
            str: The normalized question.
        """
        # Convert to lowercase
        question = question.lower()
        # Remove common words and punctuation
        common_words = {'what', 'is', 'the', 'about', 'can', 'you', 'tell', 'me', 'please', 'thank', 'thanks'}
        words = [w for w in question.split() if w not in common_words]
        return ' '.join(words)
        
    def _clean_answer(self, answer: str) -> str:
        """
        Clean the answer by removing unnecessary conversational elements.
        
        Parameters:
            answer (str): The input answer.
            
        Returns:
            str: The cleaned answer.
        """
        # Remove common conversational phrases
        conversational_phrases = [
            "Hey there!", "I've got", "let me tell you",
            "feel free to ask", "I'm here to help",
            "So what do you think?", "I'm here and ready to assist"
        ]
        for phrase in conversational_phrases:
            answer = answer.replace(phrase, "")
        
        # Clean up extra whitespace and normalize punctuation
        answer = ' '.join(answer.split())
        return answer.strip()
        
    def _ensure_cache_file(self):
        """Ensure the cache file exists, create it if it doesn't."""
        cache_path = Path(self.cache_file)
        if not cache_path.exists():
            # Create the file with an empty JSON object
            with open(cache_path, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"Created new cache file at {self.cache_file}")
        
    def _load_cache(self) -> Dict:
        """Load the cache from file if it exists, otherwise return empty dict."""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save the cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            print(f"Saved cache to {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _compute_similarity(self, question1: str, question2: str) -> float:
        """Compute cosine similarity between two questions."""
        embeddings = self.model.encode([question1, question2])
        return float(np.dot(embeddings[0], embeddings[1]) / 
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    
    def find_similar_question(self, question: str, threshold: float = 0.8) -> Tuple[str, str, float]:
        """
        Find the most similar question in the cache.
        Returns (question, answer, similarity_score) if found, None otherwise.
        """
        normalized_question = self._normalize_question(question)
        best_similarity = 0
        best_match = None
        
        for cached_q, data in self.cache.items():
            normalized_cached_q = self._normalize_question(cached_q)
            similarity = self._compute_similarity(normalized_question, normalized_cached_q)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (cached_q, data['answer'], similarity)
        
        if best_match and best_match[2] >= threshold:
            return best_match
        return None
    
    def add_question(self, question: str, answer: str):
        """Add a new question and answer to the cache."""
        # Clean and normalize the answer
        cleaned_answer = self._clean_answer(answer)
        
        # Add to cache with current timestamp
        self.cache[question] = {
            'answer': cleaned_answer,
            'timestamp': datetime.now().isoformat(),
            'normalized_question': self._normalize_question(question)
        }
        self._save_cache()

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
        # Initialize the question cache
        self.question_cache = QuestionCache()
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
          - Comprehensive content analysis and decision making.
          - Summarization.
          - Uncertainty response.
          - Answer generation.
          - Answer refinement.
          - Response formatting.

        It also sets up edges for linear progression and conditional branching based on LLM decisions.
        """
        # Add nodes to the graph with corresponding callback functions.
        self.graph.add_node("capture_input", self.input_node)
        self.graph.add_node("retrieval", self.retrieval_node)
        self.graph.add_node("content_analysis", self.content_analysis_node)
        self.graph.add_node("summarization", self.summarization_node)
        self.graph.add_node("uncertainty_response", self.uncertainty_response_node)
        self.graph.add_node("generation", self.generation_node)
        self.graph.add_node("answer_refinement", self.answer_refinement_node)
        self.graph.add_node("format_response", self.format_response_node)

        # Define linear flow from the start to input, retrieval, then analysis.
        self.graph.add_edge(START, "capture_input")
        self.graph.add_edge("capture_input", "retrieval")
        self.graph.add_edge("retrieval", "content_analysis")

        # Define conditional branching from the content analysis node
        self.graph.add_conditional_edges(
            "content_analysis",
            lambda state: state.get("next"),
            {
                "summarization": "summarization",
                "uncertainty_response": "uncertainty_response",
                "generation": "generation"
            }
        )

        # If summarization is executed, then proceed to generation
        self.graph.add_edge("summarization", "generation")

        # After generation, proceed to answer refinement
        self.graph.add_edge("generation", "answer_refinement")

        # After refinement, proceed to response formatting
        self.graph.add_edge("answer_refinement", "format_response")

        # Both uncertainty_response and format_response lead to the END node
        self.graph.add_edge("uncertainty_response", END)
        self.graph.add_edge("format_response", END)
    
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

    def content_analysis_node(self, state: State) -> dict:
        """
        Comprehensive analysis of the retrieved content to determine the next steps.
        Analyzes content for:
        1. Need for summarization
        2. Uncertainty in the information
        3. Need for refinement
        4. Overall quality and completeness

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the next node decision and analysis flags.
        """
        # Get the retrieval message
        retrieval_msg = next(
            (msg for msg in state["messages"] 
            if isinstance(msg, ToolMessage) and msg.name == "document_retriever"),
            None
        )
        
        if not retrieval_msg:
            state["next"] = "uncertainty_response"
            return state

        # Extract the actual content (remove the header)
        content = retrieval_msg.content.replace("Retrieved context:\n", "").strip()
        
        # Check content length (rough estimate of tokens)
        content_length = len(content.split())
        needs_summary = content_length > 500  # If content is longer than 500 words, summarize
        
        # Create a comprehensive prompt for the LLM to analyze the content
        prompt = f"""Analyze the following retrieved content and determine the best way to process it.
Consider all aspects and provide a structured response:

Content to analyze:
{content}

Analyze the following aspects:

1. Content Length and Complexity
   - Is the content longer than 300 words?
   - Does it contain multiple paragraphs or sections?
   - Is there redundant or repetitive information?
   - Would it benefit from being more concise?

2. Information Quality
   - Is the information complete and specific?
   - Are there any ambiguities or uncertainties?
   - Is the information relevant to the query?
   - Is there unnecessary detail that could be condensed?

3. Need for Refinement
   - Would follow-up questions help get more specific information?
   - Are there multiple aspects that could be clarified?
   - Is the information too broad or general?
   - Could the information be presented more clearly?

4. Summarization Benefits
   - Would summarizing help focus on key points?
   - Is there extraneous information that could be removed?
   - Would a shorter version be more effective?
   - Could the information be more impactful if condensed?

Respond in the following format:
UNCERTAINTY_PRESENT: [yes/no]
REFINEMENT_NEEDED: [yes/no]
NEXT_STEP: [summarization/uncertainty_response/generation]
REASONING: [brief explanation of the decision, including specific reasons for or against summarization]"""

        # Get the LLM's analysis
        response = llm.query(prompt)
        
        # Parse the response
        analysis = {}
        for line in response.content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                analysis[key.strip()] = value.strip().lower()
        
        # Determine next step based on content length and analysis
        next_step = analysis.get("next_step", "generation")
        if needs_summary and next_step != "uncertainty_response":
            next_step = "summarization"
        elif next_step == "summarization" and not needs_summary:
            # If LLM suggests summarization but content is short, check reasoning
            reasoning = analysis.get("reasoning", "").lower()
            if any(keyword in reasoning for keyword in ["redundant", "repetitive", "condense", "concise", "focus"]):
                next_step = "summarization"
            else:
                next_step = "generation"
        
        # Update state with analysis results
        state.update({
            "needs_summary": needs_summary,
            "uncertainty": analysis.get("uncertainty_present", "no") == "yes",
            "needs_refinement": analysis.get("refinement_needed", "no") == "yes",
            "next": next_step
        })
        
        return state
    
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
    
    def answer_refinement_node(self, state: State) -> dict:
        """
        Generate follow-up questions to help the user get more specific information.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with refined answer and follow-up questions.
        """
        # Get the last AI message (the generated answer)
        last_ai_msg = next(
            (msg for msg in reversed(state["messages"])
            if isinstance(msg, AIMessage)),
            None
        )
        
        if not last_ai_msg:
            return state

        # Create a prompt for the LLM to generate follow-up questions
        prompt = f"""Generate 2-3 specific follow-up questions that would help the user get more precise information
about the following answer:

Answer:
{last_ai_msg.content}

Consider:
1. What aspects of the answer could be clarified?
2. What additional information would be helpful?
3. What specific details might the user want to know more about?

Respond with just the follow-up questions, one per line. Do not include any numbering or formatting."""

        # Get the LLM's follow-up questions
        response = llm.query(prompt)
        follow_up_questions = [q.strip() for q in response.content.split("\n") if q.strip()]
        
        # Clean up the original answer content
        original_content = last_ai_msg.content
        
        # If the content is a JSON string, try to format it nicely
        if original_content.startswith('{') and original_content.endswith('}'):
            try:
                import json
                json_data = json.loads(original_content)
                formatted_json = json.dumps(json_data, indent=2)
                original_content = formatted_json
            except json.JSONDecodeError:
                pass  # If JSON parsing fails, use the original content
        
        # Create a new message with the refined answer and follow-up questions
        if follow_up_questions:
            refined_content = f"{original_content}\n\nTo help you get more specific information, here are some follow-up questions:\n"
            for i, question in enumerate(follow_up_questions, 1):
                # Remove any existing numbering from the question
                question = question.lstrip('0123456789. ')
                refined_content += f"{i}. {question}\n"
            
            # Replace the last AI message with the refined version
            state["messages"][-1] = AIMessage(content=refined_content)
        
        return state
    
    def format_response_node(self, state: State) -> dict:
        """
        Format the final response to be more conversational and natural.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the formatted response.
        """
        # Get the last AI message
        last_ai_msg = next(
            (msg for msg in reversed(state["messages"])
            if isinstance(msg, AIMessage)),
            None
        )
        
        if not last_ai_msg:
            return state

        # Create a prompt for the LLM to format the response
        prompt = f"""Convert the following response into a natural, conversational format.
Make it sound like a human speaking to another human. Remove any JSON formatting, technical structures,
or formal language. Keep the information but present it in a friendly, engaging way.
Avoid phrases like "like we humans would do" or "just like humans". Be natural and conversational.

Original response:
{last_ai_msg.content}

Guidelines:
1. Use natural, conversational language
2. Remove any JSON structures or technical formatting
3. Keep the information organized but in a flowing narrative
4. Maintain the follow-up questions but integrate them naturally
5. Use contractions and casual language where appropriate
6. Add conversational transitions between topics
7. Avoid meta-commentary about being human or conversational
8. Keep responses concise and focused

Respond with just the reformatted text."""

        # Get the LLM's formatted response
        response = llm.query(prompt)
        formatted_content = response.content.strip()
        
        # Replace the last AI message with the formatted version
        state["messages"][-1] = AIMessage(content=formatted_content)
        
        return state
    
    def run(self, initial_input: str, system_prompt: str = None) -> str:
        """
        Execute the document agent's conversation flow.

        Parameters:
            initial_input (str): The user's query to initiate the conversation.
            system_prompt (str, optional): An optional system prompt to set the conversational context.

        Returns:
            str: The cleaned output from the final AI-generated message.
        """
        # Check cache for similar questions first
        similar_question = self.question_cache.find_similar_question(initial_input)
        if similar_question:
            cached_question, cached_answer, similarity = similar_question
            # Return the cached answer directly if similarity is high enough
            if similarity > 0.8:  # Using the same threshold as find_similar_question
                return cached_answer

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

        # Configuration settings for state graph execution
        config = {"configurable": {"thread_id": "1"}}

        # Process the conversation state through the compiled graph in streaming mode.
        for step in self.compiled_graph.stream(state, config, stream_mode="values"):
            pass

        # Retrieve the final AI message
        final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
        
        if final_ai_msg:
            # Cache the question and answer
            self.question_cache.add_question(initial_input, final_ai_msg.content)
            return clean_output(final_ai_msg.content)
        
        return "No response generated."


if __name__ == "__main__":
    # Instantiate the DocumentAgent.
    agent = DocumentAgent()
    print("Chat with the DocumentAgent. Type 'exit' or 'quit' to stop.")

    # Run a loop to continuously accept user input.
    while True:
        # Read user input.
        user_input = input("You: ")
        # Allow the user to exit the loop.
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        # Run the agent with the provided input.
        response = agent.run(user_input)
        print("Agent:", response)
