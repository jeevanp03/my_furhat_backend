"""
Document Agent Module

This module implements a sophisticated document interaction system that combines RAG (Retrieval-Augmented Generation)
with a state-based conversation flow. The DocumentAgent orchestrates a multi-step workflow for processing and
responding to document-related queries.

Key Components:
    - DocumentAgent: Main class managing the conversation workflow
    - QuestionCache: Caching system for questions and answers with similarity matching
    - State Graph: Manages conversation flow and state transitions

The workflow includes:
    1. Input processing
    2. Document retrieval
    3. Content summarization
    4. Uncertainty checking
    5. Response generation
    6. Follow-up handling
"""

import os
import logging
import uuid
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from typing_extensions import TypedDict, Annotated, List
from my_furhat_backend.config.settings import config
from langgraph.checkpoint.memory import MemorySaver
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
from transformers import pipeline

from my_furhat_backend.models.chatbot_factory import create_chatbot
from my_furhat_backend.utils.util import clean_output, summarize_text
from my_furhat_backend.RAG.rag_flow import RAG
from my_furhat_backend.utils.gpu_utils import print_gpu_status, clear_gpu_cache

# Initialize RAG with caching
rag_instance = RAG(
    hf=True,
    persist_directory=config["VECTOR_STORE_PATH"],
    path_to_document=os.path.join(config["DOCUMENTS_PATH"], "NorwAi annual report 2023.pdf")
)

# Initialize chatbot with optimized settings
chatbot = create_chatbot(
    "llama",
    n_ctx=4096,
    n_batch=512,
    n_threads=4,
    n_gpu_layers=32
)
llm = chatbot.llm

class State(TypedDict):
    """
    Conversation state type definition.
    
    Attributes:
        messages (List[BaseMessage]): List of conversation messages
        input (str): Current user input
    """
    messages: Annotated[List[BaseMessage], "add_messages"]
    input: str

class QuestionCache:
    """
    Manages caching of questions and answers with similarity-based retrieval.
    
    This class provides functionality to:
    - Cache questions and answers
    - Find similar questions using semantic similarity
    - Normalize and clean questions/answers
    - Persist cache to disk
    
    Attributes:
        cache_file (str): Path to the cache file
        cache (Dict): In-memory cache of questions and answers
        model (SentenceTransformer): Model for computing semantic similarity
    """
    
    def __init__(self, cache_file: str = os.path.join(config["TRANSFORMERS_CACHE"], "question_cache.json")):
        """
        Initialize the QuestionCache.
        
        Args:
            cache_file (str): Path to the cache file
        """
        self.cache_file = cache_file
        self._ensure_cache_file()
        self.cache: Dict[str, Dict] = self._load_cache()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _normalize_question(self, question: str) -> str:
        """
        Normalize the question by removing common words and standardizing format.
        
        Args:
            question (str): The input question
            
        Returns:
            str: The normalized question
        """
        question = question.lower()
        common_words = {'what', 'is', 'the', 'about', 'can', 'you', 'tell', 'me', 'please', 'thank', 'thanks'}
        words = [w for w in question.split() if w not in common_words]
        return ' '.join(words)
        
    def _clean_answer(self, answer: str) -> str:
        """
        Clean the answer by removing unnecessary conversational elements.
        
        Args:
            answer (str): The input answer
            
        Returns:
            str: The cleaned answer
        """
        conversational_phrases = [
            "Hey there!", "I've got", "let me tell you",
            "feel free to ask", "I'm here to help",
            "So what do you think?", "I'm here and ready to assist"
        ]
        for phrase in conversational_phrases:
            answer = answer.replace(phrase, "")
        
        answer = ' '.join(answer.split())
        return answer.strip()
        
    def _ensure_cache_file(self) -> None:
        """Ensure the cache file exists, create it if it doesn't."""
        cache_path = Path(self.cache_file)
        if not cache_path.exists():
            with open(cache_path, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"Created new cache file at {self.cache_file}")
        
    def _load_cache(self) -> Dict:
        """
        Load the cache from file if it exists, otherwise return empty dict.
        
        Returns:
            Dict: The loaded cache or empty dict if loading fails
        """
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading cache: {e}")
            return {}
    
    def _save_cache(self) -> None:
        """Save the cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            print(f"Saved cache to {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _compute_similarity(self, question1: str, question2: str) -> float:
        """
        Compute cosine similarity between two questions.
        
        Args:
            question1 (str): First question
            question2 (str): Second question
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        embeddings = self.model.encode([question1, question2])
        return float(np.dot(embeddings[0], embeddings[1]) / 
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    
    def find_similar_question(self, question: str, threshold: float = 0.8) -> Optional[Tuple[str, str, float]]:
        """
        Find the most similar question in the cache.
        
        Args:
            question (str): The question to find similar matches for
            threshold (float): Minimum similarity score (0-1) to consider a match
            
        Returns:
            Optional[Tuple[str, str, float]]: Tuple of (question, answer, similarity) if found,
                None if no match above threshold
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
    
    def add_question(self, question: str, answer: str) -> None:
        """
        Add a new question and answer to the cache.
        
        Args:
            question (str): The question to cache
            answer (str): The answer to cache
        """
        cleaned_answer = self._clean_answer(answer)
        
        self.cache[question] = {
            'answer': cleaned_answer,
            'timestamp': datetime.now().isoformat(),
            'normalized_question': self._normalize_question(question)
        }
        self._save_cache()

class DocumentAgent:
    """
    Orchestrates a multi-step conversational workflow for document interaction.
    
    The agent manages:
    - Document retrieval and context gathering
    - Content summarization and analysis
    - Uncertainty checking and clarification
    - Response generation and follow-up handling
    - Conversation state management
    
    The workflow is implemented as a state graph with checkpointed memory for resumption.
    """
    
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the DocumentAgent.
        
        Args:
            model_id (str): ID of the model to use for the chatbot
        """
        print_gpu_status()
        
        self.memory = MemorySaver()
        
        self.rag_instance = RAG(
            hf=True,
            persist_directory=config["VECTOR_STORE_PATH"],
            path_to_document=os.path.join(config["DOCUMENTS_PATH"], "NorwAi annual report 2023.pdf")
        )
        
        self.chatbot = create_chatbot(
            "llama",
            model_id=model_id,
            n_ctx=4096,
            n_batch=512,
            n_threads=4,
            n_gpu_layers=32
        )
        self.llm = self.chatbot.llm
        
        print_gpu_status()
        
        self.graph = StateGraph(State)
        
        self.question_cache = QuestionCache()
        self.context_cache = {}
        self.summary_cache = {}
        
        self._build_graph()
        
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        self.conversation_memory = []
        self.max_memory_size = 10
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device="mps",
            model_kwargs={"cache_dir": "my_furhat_backend/cache"}  # Enable model caching
        )
        
        # Initialize personality traits
        self.personality_traits = {
            "curiosity": 0.8,
            "empathy": 0.7,
            "enthusiasm": 0.6
        }
    
    def __del__(self):
        """Cleanup method to clear GPU cache when the agent is destroyed."""
        clear_gpu_cache()
    
    def _build_graph(self) -> None:
        """
        Build the state graph defining the conversation workflow.
        
        The graph consists of the following nodes:
        1. input_node: Process user input
        2. retrieval_node: Retrieve relevant document context
        3. summarization_node: Summarize retrieved content
        4. content_analysis_node: Analyze content for uncertainty
        5. uncertainty_response_node: Handle uncertainty cases
        6. generation_node: Generate final response
        7. format_response_node: Format and clean response
        8. answer_followup_node: Handle follow-up questions
        """
        self.graph.add_node("input", self.input_node)
        self.graph.add_node("retrieval", self.retrieval_node)
        self.graph.add_node("summarization", self.summarization_node)
        self.graph.add_node("content_analysis", self.content_analysis_node)
        self.graph.add_node("uncertainty_response", self.uncertainty_response_node)
        self.graph.add_node("generation", self.generation_node)
        self.graph.add_node("format_response", self.format_response_node)
        self.graph.add_node("answer_followup", self.answer_followup_node)
        
        # Define edges and conditions
        self.graph.add_edge(START, "input")
        self.graph.add_edge("input", "retrieval")
        self.graph.add_edge("retrieval", "summarization")
        self.graph.add_edge("summarization", "content_analysis")
        self.graph.add_edge("content_analysis", "uncertainty_response")
        self.graph.add_edge("uncertainty_response", "generation")
        self.graph.add_edge("generation", "format_response")
        self.graph.add_edge("format_response", END)
        
        # Add conditional edges for follow-up handling
        self.graph.add_conditional_edges(
            "format_response",
            self._determine_next_node,
            {
                "answer_followup": "answer_followup",
                END: END
            }
        )
        self.graph.add_edge("answer_followup", "retrieval")
        
    def input_node(self, state: State) -> dict:
        """
        Process the initial user input.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with processed input
        """
        messages = state["messages"]
        input_text = state["input"]
        
        # Add user message to conversation
        messages.append(HumanMessage(content=input_text))
        
        return {"messages": messages}
        
    def retrieval_node(self, state: State) -> dict:
        """
        Retrieve relevant document context using RAG.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with retrieved context
        """
        messages = state["messages"]
        input_text = state["input"]
        
        # Check cache for similar questions
        cached_result = self.question_cache.find_similar_question(input_text)
        if cached_result:
            cached_question, cached_answer, similarity = cached_result
            messages.append(AIMessage(content=cached_answer))
            return {"messages": messages}
            
        # Retrieve context from RAG
        context = self.rag_instance.get_relevant_context(input_text)
        self.context_cache[input_text] = context
        
        return {"messages": messages, "context": context}
        
    def summarization_node(self, state: State) -> dict:
        """
        Summarize the retrieved document context.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with summarized context
        """
        messages = state["messages"]
        context = state.get("context", "")
        
        # Check cache for existing summary
        if context in self.summary_cache:
            summary = self.summary_cache[context]
        else:
            summary = summarize_text(context)
            self.summary_cache[context] = summary
            
        return {"messages": messages, "summary": summary}
        
    def content_analysis_node(self, state: State) -> dict:
        """
        Analyze content for uncertainty and quality.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with analysis results
        """
        messages = state["messages"]
        summary = state.get("summary", "")
        
        # Analyze content quality and uncertainty
        uncertainty_score = self._analyze_uncertainty(summary)
        quality_score = self._analyze_quality(summary)
        
        return {
            "messages": messages,
            "uncertainty_score": uncertainty_score,
            "quality_score": quality_score
        }
        
    def uncertainty_response_node(self, state: State) -> dict:
        """
        Handle cases where content uncertainty is detected.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with clarification if needed
        """
        messages = state["messages"]
        uncertainty_score = state.get("uncertainty_score", 0)
        
        if uncertainty_score > 0.7:
            clarification = self._generate_clarification(state)
            messages.append(AIMessage(content=clarification))
            
        return {"messages": messages}
        
    def generation_node(self, state: State) -> dict:
        """
        Generate the final response using the chatbot.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with generated response
        """
        messages = state["messages"]
        summary = state.get("summary", "")
        
        response = self.chatbot.generate_response(messages, summary)
        messages.append(AIMessage(content=response))
        
        return {"messages": messages}
        
    def format_response_node(self, state: State) -> dict:
        """
        Format and clean the generated response.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with formatted response
        """
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Clean and format the response
        cleaned_response = clean_output(last_message)
        messages[-1].content = cleaned_response
        
        # Update conversation memory
        self._update_conversation_memory(state["input"], cleaned_response)
        
        return {"messages": messages}
        
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        result = self.sentiment_analyzer(text)[0]
        return float(result["score"]) if result["label"] == "POSITIVE" else -float(result["score"])
        
    def _adjust_tone(self, text: str, sentiment: float) -> str:
        """
        Adjust the tone of the text based on sentiment.
        
        Args:
            text (str): Text to adjust
            sentiment (float): Sentiment score
            
        Returns:
            str: Adjusted text
        """
        if sentiment < -0.5:
            return f"I understand your concern. {text}"
        elif sentiment > 0.5:
            return f"I'm glad you're interested! {text}"
        return text
        
    def _generate_engaging_prompt(self, document_name: str, answer: str) -> str:
        """
        Generate an engaging follow-up prompt.
        
        Args:
            document_name (str): Name of the document
            answer (str): Previous answer
            
        Returns:
            str: Engaging follow-up prompt
        """
        return f"Based on the {document_name}, {answer}\n\nWould you like to know more about any specific aspect?"
        
    def _update_conversation_memory(self, question: str, answer: str) -> None:
        """
        Update the conversation memory with new Q&A pair.
        
        Args:
            question (str): User question
            answer (str): System answer
        """
        self.conversation_memory.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep memory size within limit
        if len(self.conversation_memory) > self.max_memory_size:
            self.conversation_memory.pop(0)
            
    def _get_conversation_context(self) -> str:
        """
        Get the current conversation context.
        
        Returns:
            str: Formatted conversation context
        """
        context = []
        for qa in self.conversation_memory[-3:]:  # Last 3 exchanges
            context.append(f"Q: {qa['question']}\nA: {qa['answer']}")
        return "\n\n".join(context)
        
    def run(self, initial_input: str, system_prompt: str = None) -> str:
        """
        Run the conversation workflow.
        
        Args:
            initial_input (str): Initial user input
            system_prompt (str, optional): System prompt to guide the conversation
            
        Returns:
            str: Final response
        """
        # Initialize state
        state = {
            "messages": [],
            "input": initial_input
        }
        
        # Add system prompt if provided
        if system_prompt:
            state["messages"].append(SystemMessage(content=system_prompt))
            
        # Run the graph
        final_state = self.compiled_graph.invoke(state)
        
        # Return the last message
        return final_state["messages"][-1].content
        
    def _answer_follow_up(self, follow_up_question: str) -> str:
        """
        Handle follow-up questions using conversation context.
        
        Args:
            follow_up_question (str): Follow-up question
            
        Returns:
            str: Response to follow-up question
        """
        # Get conversation context
        context = self._get_conversation_context()
        
        # Generate response using context
        messages = [
            SystemMessage(content="You are a helpful assistant answering follow-up questions."),
            HumanMessage(content=f"Previous conversation:\n{context}\n\nFollow-up question: {follow_up_question}")
        ]
        
        response = self.chatbot.generate_response(messages)
        return clean_output(response)
        
    def engage(self, document_name: str, answer: str) -> str:
        """
        Generate an engaging follow-up prompt.
        
        Args:
            document_name (str): Name of the document
            answer (str): Previous answer
            
        Returns:
            str: Engaging follow-up prompt
        """
        return self._generate_engaging_prompt(document_name, answer)
        
    def answer_followup_node(self, state: State) -> dict:
        """
        Process follow-up questions.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with follow-up response
        """
        messages = state["messages"]
        input_text = state["input"]
        
        response = self._answer_follow_up(input_text)
        messages.append(AIMessage(content=response))
        
        return {"messages": messages}
        
    def _determine_next_node(self, state: State) -> str:
        """
        Determine the next node in the conversation flow.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            str: Next node name
        """
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Check if this is a follow-up question
        if "follow-up" in last_message.lower() or "more about" in last_message.lower():
            return "answer_followup"
            
        return END

if __name__ == "__main__":
    # Instantiate the DocumentAgent.
    agent = DocumentAgent()
    print("Chat with the DocumentAgent. Type 'exit' or 'quit' to stop.")
    print("GPU Status:")
    print_gpu_status()

    # Run a loop to continuously accept user input.
    while True:
        # Read user input.
        user_input = input("You: ")
        # Allow the user to exit the loop.
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        try:
            # Run the agent with the provided input.
            print("\nProcessing your query...")
            response = agent.run(user_input)
            print("\nAgent:", response)
            
            # Test the engage functionality
            print("\nGenerating follow-up...")
            follow_up = agent.engage("CMRPublished", response)
            print("Agent Follow-up:", follow_up)
            
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.\n")
