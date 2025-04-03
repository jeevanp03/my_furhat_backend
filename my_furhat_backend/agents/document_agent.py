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
import shutil
import sys
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
import re
import torch

from my_furhat_backend.models.chatbot_factory import create_chatbot
from my_furhat_backend.utils.util import clean_output, summarize_text
from my_furhat_backend.RAG.rag_flow import RAG
from my_furhat_backend.utils.gpu_utils import print_gpu_status, clear_gpu_cache
from my_furhat_backend.models.llm_factory import create_llm

# Set up cache directories
CACHE_DIR = config["HF_HOME"]
os.makedirs(CACHE_DIR, exist_ok=True)

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
    
    def __init__(self, cache_file: str = os.path.join(config["HF_HOME"], "question_cache.json")):
        """
        Initialize the QuestionCache.
        
        Args:
            cache_file (str): Path to the cache file
        """
        self.cache_file = cache_file
        self._ensure_cache_file()
        self.cache: Dict[str, Dict] = self._load_cache()
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=config["HF_HOME"])
        
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
    
    def __init__(self, model_id: str = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"):
        """
        Initialize the DocumentAgent.

        Args:
            model_id (str): ID of the model to use for the chatbot
        """
        print_gpu_status()
        
        self.memory = MemorySaver()
        
        # Initialize RAG with caching
        self.rag_instance = RAG(
            hf=True,
            persist_directory=config["VECTOR_STORE_PATH"],
            path_to_document=os.path.join(config["DOCUMENTS_PATH"], "NorwAi annual report 2023.pdf")
        )
        
        # Initialize chatbot with optimized settings
        # Only pass the model_id and essential parameters
        self.chatbot = create_chatbot(
            "llama",
            model_id=model_id,
            n_ctx=4096,  # Reduced context window
            n_batch=512,  # Increased batch size
            n_gpu_layers=32  # Use more GPU layers
        )
        self.llm = self.chatbot.llm
        
        print_gpu_status()
        
        self.graph = StateGraph(State)
        
        # Initialize caches with larger sizes
        self.question_cache = QuestionCache()
        self.context_cache = {}
        self.summary_cache = {}  # New cache for document summaries
        
        self._build_graph()
        
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        # Initialize conversation memory with a larger size
        self.conversation_memory = []
        self.max_memory_size = 10  # Increased from 5
        
        # Initialize sentiment analyzer with specific model and caching
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device="cuda",  # Specify device directly
                model_kwargs={"cache_dir": config["HF_HOME"]}  # Enable model caching
            )
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            self.sentiment_analyzer = None
        
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
        # Add nodes to the graph with corresponding callback functions
        self.graph.add_node("capture_input", self.input_node)
        self.graph.add_node("retrieval", self.retrieval_node)
        self.graph.add_node("content_analysis", self.content_analysis_node)
        self.graph.add_node("summarization", self.summarization_node)
        self.graph.add_node("uncertainty_response", self.uncertainty_response_node)
        self.graph.add_node("generation", self.generation_node)
        self.graph.add_node("format_response", self.format_response_node)
        self.graph.add_node("answer_followup", self.answer_followup_node)

        # Define linear flow from the start to input
        self.graph.add_edge(START, "capture_input")

        # Add conditional edge from input to either retrieval or answer_followup
        self.graph.add_conditional_edges(
            "capture_input",
            lambda state: self._determine_next_node(state),
            {
                "retrieval": "retrieval",
                "answer_followup": "answer_followup"
            }
        )

        # Define linear flow from retrieval to analysis
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

        # After generation, proceed directly to response formatting
        self.graph.add_edge("generation", "format_response")

        # Both uncertainty_response and format_response lead to the END node
        self.graph.add_edge("uncertainty_response", END)
        self.graph.add_edge("format_response", END)
        self.graph.add_edge("answer_followup", END)
    
    def input_node(self, state: State) -> dict:
        """
        Process the initial user input.

        Args:
            state (State): Current conversation state

        Returns:
            dict: Updated state with processed input
        """
        # Ensure that the 'messages' list exists in the state
        state.setdefault("messages", [])
        # Create a HumanMessage using the user's input
        human_msg = HumanMessage(content=state.get("input", ""))
        # Append the human message to the conversation history
        state["messages"].append(human_msg)
        return {"messages": state["messages"]}
    
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
        context = self.rag_instance.get_document_context(input_text)
        self.context_cache[input_text] = context
        
        # Create a ToolMessage with the retrieval results
        tool_msg = ToolMessage(
            content=f"Retrieved context:\n{context}",
            name="document_retriever",
            tool_call_id=str(uuid.uuid4())
        )
        messages.append(tool_msg)
        
        return {"messages": messages}
    
    def summarization_node(self, state: State) -> dict:
        """
        Summarize the retrieved document context.

        Args:
            state (State): Current conversation state

        Returns:
            dict: Updated state with summarized context
        """
        messages = state["messages"]
        
        # Locate the ToolMessage that holds the retrieved document context
        retrieval_msg = next(
            (msg for msg in messages
             if isinstance(msg, ToolMessage) and msg.name == "document_retriever"),
            None
        )
        
        if retrieval_msg:
            # Remove any header text from the retrieval message
            text_to_summarize = retrieval_msg.content.replace("Retrieved context:\n", "")
            
            # Check if we have a cached summary for this content
            content_hash = hash(text_to_summarize)
            if content_hash in self.summary_cache:
                summarized_text = self.summary_cache[content_hash]
            else:
                # Generate a summary of the retrieved content
                summarized_text = summarize_text(
                    text_to_summarize,
                    max_length=400,
                    min_length=50
                )
                # Cache the summary
                self.summary_cache[content_hash] = summarized_text
                
                # Limit cache size
                if len(self.summary_cache) > 1000:
                    self.summary_cache.pop(next(iter(self.summary_cache)))
        else:
            summarized_text = "No document context available to summarize."
        
        # Create a new ToolMessage for the summarized context
        summary_msg = ToolMessage(
            content=f"Summarized context:\n{summarized_text}",
            name="summarizer",
            tool_call_id=str(uuid.uuid4())
        )
        messages.append(summary_msg)
        
        return {"messages": messages}
        
    def content_analysis_node(self, state: State) -> dict:
        """
        Analyze content for uncertainty and quality.
        
        Args:
            state (State): Current conversation state

        Returns:
            dict: Updated state with analysis results
        """
        messages = state["messages"]
        
        # Get the retrieval message
        retrieval_msg = next(
            (msg for msg in messages 
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

3. Summarization Benefits
   - Would summarizing help focus on key points?
   - Is there extraneous information that could be removed?
   - Would a shorter version be more effective?
   - Could the information be more impactful if condensed?

Respond in the following format:
UNCERTAINTY_PRESENT: [yes/no]
NEXT_STEP: [summarization/uncertainty_response/generation]
REASONING: [brief explanation of the decision, including specific reasons for or against summarization]"""

        # Get the LLM's analysis
        response = self.llm.query(prompt)
        
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
            "next": next_step
        })
        
        return state
    
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
        messages = state.get("messages", [])
        summary = state.get("summary", "")
        
        # Add the summary as context to the conversation
        if summary:
            messages.insert(0, SystemMessage(content=f"Context: {summary}"))
        
        # Use the chatbot to generate the response
        updated_state = self.chatbot.chatbot({"messages": messages})
        return updated_state
        
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
        
        # Split into sentences and analyze content
        sentences = cleaned_response.split('.')
        
        # Determine response type and adjust accordingly
        response_type = self._analyze_response_type(cleaned_response)
        
        if response_type == "technical":
            # For technical content, keep it clear but friendly
            if len(sentences) > 2:
                cleaned_response = '. '.join(sentences[:2]) + '.'
            if not any(cleaned_response.lower().startswith(word) for word in ['well', 'so', 'actually', 'you know', 'i mean']):
                cleaned_response = f"Well, {cleaned_response.lower()}"
        elif response_type == "casual":
            # For casual content, be more conversational
            if len(sentences) > 3:
                cleaned_response = '. '.join(sentences[:3]) + '.'
            if not any(cleaned_response.lower().startswith(word) for word in ['well', 'so', 'actually', 'you know', 'i mean']):
                cleaned_response = f"So, {cleaned_response.lower()}"
        else:
            # For general content, keep it balanced
            if len(sentences) > 2:
                cleaned_response = '. '.join(sentences[:2]) + '.'
            if not any(cleaned_response.lower().startswith(word) for word in ['well', 'so', 'actually', 'you know', 'i mean']):
                cleaned_response = f"Well, {cleaned_response.lower()}"
        
        # Remove common repetitive phrases
        cleaned_response = re.sub(r'Let me think about that\.?\s*', '', cleaned_response)
        cleaned_response = re.sub(r'I notice you\'re interested in\.?\s*', '', cleaned_response)
        cleaned_response = re.sub(r'Based on the document\.?\s*', '', cleaned_response)
        cleaned_response = re.sub(r'According to the document\.?\s*', '', cleaned_response)
        cleaned_response = re.sub(r'In the document\.?\s*', '', cleaned_response)
        
        # Add natural conversational elements based on content
        if any(word in cleaned_response.lower() for word in ['interesting', 'fascinating', 'surprising']):
            cleaned_response = f"That's fascinating! {cleaned_response.lower()}"
        elif any(word in cleaned_response.lower() for word in ['important', 'significant', 'crucial']):
            cleaned_response = f"Actually, {cleaned_response.lower()}"
        elif any(word in cleaned_response.lower() for word in ['complex', 'complicated', 'difficult']):
            cleaned_response = f"You know, {cleaned_response.lower()}"
        
        messages[-1].content = cleaned_response
        
        # Update conversation memory
        self._update_conversation_memory(state["input"], cleaned_response)
        
        return {"messages": messages}
        
    def _analyze_response_type(self, text: str) -> str:
        """
        Analyze the type of response to determine appropriate formatting.
        
        Args:
            text (str): The response text to analyze
            
        Returns:
            str: The type of response ('technical', 'casual', or 'general')
        """
        # Technical indicators
        technical_words = {'implementation', 'algorithm', 'process', 'system', 'method', 'function', 'data', 'analysis'}
        # Casual indicators
        casual_words = {'cool', 'awesome', 'interesting', 'fun', 'great', 'nice', 'good', 'bad', 'wow'}
        
        words = set(text.lower().split())
        technical_count = len(words.intersection(technical_words))
        casual_count = len(words.intersection(casual_words))
        
        if technical_count > casual_count and technical_count > 2:
            return "technical"
        elif casual_count > technical_count and casual_count > 2:
            return "casual"
        else:
            return "general"
        
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
        # Use personality traits to influence prompt generation
        curiosity_level = self.personality_traits["curiosity"]
        empathy_level = self.personality_traits["empathy"]
        
        # Generate different types of prompts based on personality
        if curiosity_level > 0.7:
            return f"I'm really curious about this! What would you like to explore next about {document_name}?"
        elif empathy_level > 0.7:
            return f"I find this topic fascinating. What aspects of {document_name} would you like to discuss further?"
        else:
            return f"Would you like to know more about {document_name}?"
            
    def _update_conversation_memory(self, question: str, answer: str) -> None:
        """
        Update the conversation memory with new Q&A pair.
        
        Args:
            question (str): User question
            answer (str): System answer
        """
        # Add new exchange
        self.conversation_memory.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last N exchanges for context
        if len(self.conversation_memory) > self.max_memory_size:
            # Remove oldest entries
            self.conversation_memory = self.conversation_memory[-self.max_memory_size:]
            
        # Clean up old follow-up questions
        self.conversation_memory = [
            msg for msg in self.conversation_memory 
            if not (msg.get("follow_up", False) and 
                   (datetime.now() - datetime.fromisoformat(msg["timestamp"])).days > 1)
        ]
            
    def _get_conversation_context(self) -> str:
        """
        Get a summary of the conversation context.

        Returns:
            str: Formatted conversation context
        """
        if not self.conversation_memory:
            return ""
            
        context = "Previous conversation:\n"
        for exchange in self.conversation_memory:
            context += f"Q: {exchange['question']}\nA: {exchange['answer']}\n"
        return context
    
    def clear_all_caches(self) -> None:
        """
        Clear all caches including question cache, context cache, and GPU cache.
        """
        # Clear question cache file
        if os.path.exists(self.question_cache.cache_file):
            os.remove(self.question_cache.cache_file)
            self.question_cache.cache = {}
            print("Question cache cleared")
        
        # Clear context cache
        self.context_cache.clear()
        print("Context cache cleared")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        
        # Clear conversation memory
        self.conversation_memory.clear()
        print("Conversation memory cleared")
        
        # Clear summary cache
        self.summary_cache.clear()
        print("Summary cache cleared")

    def run(self, initial_input: str, system_prompt: str = None) -> str:
        """
        Execute the document agent's conversation flow.

        Args:
            initial_input (str): The user's query to initiate the conversation
            system_prompt (str, optional): An optional system prompt to set the conversational context

        Returns:
            str: The cleaned output from the final AI-generated message
        """
        try:
            # Truncate input if too long
            max_input_length = 500  # words
            input_words = initial_input.split()
            if len(input_words) > max_input_length:
                initial_input = ' '.join(input_words[:max_input_length]) + '...'

            # Don't cache or process "I don't know" type responses
            if any(phrase in initial_input.lower() for phrase in ["i don't know", "i do not know", "you tell me", "tell me"]):
                # Get the last follow-up question from the conversation
                last_follow_up = next(
                    (msg for msg in reversed(self.conversation_memory)
                    if "follow_up" in msg),
                    None
                )
                if last_follow_up:
                    # Answer the follow-up question instead of treating it as a new query
                    return self._answer_follow_up(last_follow_up["question"])
                return "I apologize, but I don't have enough context to provide a meaningful answer."

            # Check cache for similar questions first
            similar_question = self.question_cache.find_similar_question(initial_input)
            if similar_question:
                cached_question, cached_answer, similarity = similar_question
                # Return the cached answer directly if similarity is high enough
                if similarity > 0.8:  # Using the same threshold as find_similar_question
                    return cached_answer

            # Use a default system prompt if none is provided
            if system_prompt is None:
                system_prompt = (
                    "You are a friendly and knowledgeable assistant having a casual conversation. "
                    "Keep your responses concise and engaging - aim for 2-3 sentences maximum. "
                    "Use natural, conversational language and avoid formal or academic tone. "
                    "Focus on the most interesting or relevant aspects of the topic. "
                    "Don't repeat information unless necessary. "
                    "If the user seems disengaged, be more concise. "
                    "If they show interest, you can elaborate slightly. "
                    "Use contractions and casual expressions. "
                    "Avoid phrases like 'Let me think about that' or 'I notice you're interested in'. "
                    "Be direct and engaging, like chatting with a friend. "
                    "Adapt your tone based on the user's engagement level. "
                    "If they ask short questions, give short answers. "
                    "If they ask detailed questions, provide more context. "
                    "Use natural transitions between topics. "
                    "Avoid robotic or overly formal language. "
                    "Be conversational but professional. "
                    "Use appropriate humor when relevant. "
                    "Show enthusiasm for interesting topics. "
                    "Be empathetic when discussing complex or challenging topics."
                )

            # Initialize the conversation state with the system prompt as the first human message
            state: State = {
                "input": initial_input,
                "messages": [HumanMessage(content=system_prompt)]
            }

            # Configuration settings for state graph execution
            config = {"configurable": {"thread_id": "1"}}

            # Process the conversation state through the compiled graph in streaming mode
            for step in self.compiled_graph.stream(state, config, stream_mode="values"):
                pass

            # Retrieve the final AI message
            final_ai_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
            
            if final_ai_msg:
                # Truncate the response if too long
                max_response_length = 1000  # words
                response_words = final_ai_msg.content.split()
                if len(response_words) > max_response_length:
                    final_ai_msg.content = ' '.join(response_words[:max_response_length]) + '...'
                
                # Cache the question and answer
                self.question_cache.add_question(initial_input, final_ai_msg.content)
                # Store in conversation memory with document name
                self.conversation_memory.append({
                    "question": initial_input,
                    "answer": final_ai_msg.content,
                    "document_name": "CMRPublished",  # Default document name
                    "timestamp": datetime.now().isoformat()
                })
                return clean_output(final_ai_msg.content)
            
            return "No response generated."
            
        except Exception as e:
            if "exceed context window" in str(e):
                # Clear caches and try again with a shorter input
                self.clear_all_caches()
                return "I apologize, but the question was too long. Could you please rephrase it to be more concise?"
            else:
                # For other errors, clear caches and return error message
                self.clear_all_caches()
                return f"I encountered an error: {str(e)}. The caches have been cleared. Please try again."

    def _answer_follow_up(self, follow_up_question: str) -> str:
        """
        Answer a follow-up question directly without treating it as a new query.
        
        Args:
            follow_up_question (str): The follow-up question to answer
            
        Returns:
            str: The answer to the follow-up question
        """
        # Get the previous answer from conversation memory
        previous_exchange = next(
            (msg for msg in reversed(self.conversation_memory)
            if not msg.get("follow_up", False)),  # Get the last non-follow-up exchange
            None
        )
        
        if not previous_exchange:
            return "I apologize, but I don't have enough context to provide a meaningful answer."
            
        previous_answer = previous_exchange.get("answer", "")
        
        # Truncate previous answer to prevent token overflow
        max_answer_length = 200   # words
        answer_words = previous_answer.split()
        
        if len(answer_words) > max_answer_length:
            previous_answer = ' '.join(answer_words[:max_answer_length]) + '...'
        
        # Create a prompt to answer the follow-up question
        prompt = f"""Answer the following follow-up question based on the previous answer.

Previous Answer:
{previous_answer}

Follow-up Question:
{follow_up_question}

Guidelines:
1. Provide a direct answer to the follow-up question
2. Use the previous answer to support your response
3. Keep the response concise and focused
4. Use natural, conversational language
5. If you can't answer based on the previous context, say so clearly
6. Make sure your answer builds on the previous discussion
7. Avoid repeating information from the previous answer unless relevant to the follow-up

Generate a direct answer:"""

        # Get the answer from the LLM
        response = self.llm.query(prompt)
        answer = response.content if isinstance(response, AIMessage) else str(response)
        
        return clean_output(answer)

    def engage(self, document_name: str, answer: str) -> str:
        """
        Generate an engaging follow-up question based on the document context and previous answer.
        
        Args:
            document_name (str): Name of the document being discussed
            answer (str): The previous answer to generate a follow-up for
            
        Returns:
            str: A conversational follow-up question
        """
        # Get document context from cache or retrieve it
        if document_name not in self.context_cache:
            self.context_cache[document_name] = self.rag_instance.get_document_context(document_name)
        
        # Extract text content from document list and limit context length
        context_docs = self.context_cache[document_name]
        context = "\n".join(doc.page_content for doc in context_docs)
        
        # Truncate context and answer to prevent token overflow
        max_context_length = 1000  # words
        max_answer_length = 500    # words
        
        # Split into words and truncate
        context_words = context.split()
        answer_words = answer.split()
        
        if len(context_words) > max_context_length:
            context = ' '.join(context_words[:max_context_length]) + '...'
        if len(answer_words) > max_answer_length:
            answer = ' '.join(answer_words[:max_answer_length]) + '...'
        
        # Create a more sophisticated prompt for generating engaging follow-ups
        prompt = f"""Based on the previous answer and document context, generate a natural, engaging follow-up question.

Previous Answer:
{answer}

Document Context:
{context}

Guidelines for generating an engaging follow-up:
1. Focus on the most interesting or surprising aspect of the previous answer
2. Ask about implications, consequences, or future developments
3. Use natural, conversational language that a human would use
4. Make the question specific and focused
5. Avoid generic or vague questions
6. Don't ask about the user's experience or opinions
7. Don't use academic or formal language
8. Don't ask yes/no questions
9. Don't repeat information from the previous answer
10. Make the question thought-provoking but not too complex
11. Use a friendly, curious tone
12. Focus on the "why" or "how" rather than just the "what"
13. Avoid questions that could be answered with a simple fact
14. Make it feel like a natural continuation of the conversation
15. Consider the document context to ensure the question is relevant
16. Keep the question concise and direct
17. Avoid using phrases like "Let me know" or "Tell me about"
18. Don't use multiple questions in one prompt
19. Don't ask for examples unless specifically relevant
20. Don't ask about personal experiences or preferences

Generate a single, engaging follow-up question:"""
        
        try:
            # Get the follow-up question from the LLM
            response = self.llm.query(prompt)
            follow_up = response.content if isinstance(response, AIMessage) else str(response)
            
            # Clean up the response to make it more conversational
            follow_up = re.sub(r'\d+\)\s*', '', follow_up)  # Remove numbered questions
            follow_up = re.sub(r'Could you elaborate on|What specific|How does|In what ways', '', follow_up)
            follow_up = re.sub(r'feel free to ask me follow up questions like:', '', follow_up)
            follow_up = re.sub(r'questions like:', '', follow_up)
            follow_up = re.sub(r'questions such as:', '', follow_up)
            follow_up = re.sub(r'like:', '', follow_up)
            follow_up = re.sub(r'such as:', '', follow_up)
            follow_up = re.sub(r'for example:', '', follow_up)
            follow_up = re.sub(r'including:', '', follow_up)
            follow_up = re.sub(r'like', '', follow_up)
            follow_up = re.sub(r'such as', '', follow_up)
            follow_up = re.sub(r'for example', '', follow_up)
            follow_up = re.sub(r'including', '', follow_up)
            follow_up = re.sub(r'etc\.', '', follow_up)
            follow_up = re.sub(r'etc', '', follow_up)
            follow_up = re.sub(r'\.\.\.', '', follow_up)
            follow_up = re.sub(r'\.\.', '', follow_up)
            follow_up = re.sub(r'\.', '', follow_up)
            follow_up = re.sub(r'\?', '', follow_up)
            follow_up = re.sub(r'\s+', ' ', follow_up).strip()
            
            # Add a conversational prefix based on the content and personality traits
            curiosity_level = self.personality_traits["curiosity"]
            empathy_level = self.personality_traits["empathy"]
            
            if any(word in follow_up.lower() for word in ['interesting', 'fascinating', 'surprising']):
                if curiosity_level > 0.7:
                    follow_up = f"That's fascinating! {follow_up}?"
                else:
                    follow_up = f"Well, {follow_up}?"
            elif any(word in follow_up.lower() for word in ['implication', 'consequence', 'impact']):
                if empathy_level > 0.7:
                    follow_up = f"I'm curious about the implications - {follow_up}?"
                else:
                    follow_up = f"So, {follow_up}?"
            elif any(word in follow_up.lower() for word in ['future', 'develop', 'next']):
                follow_up = f"Looking ahead, {follow_up}?"
            else:
                if curiosity_level > 0.7:
                    follow_up = f"I'm curious, {follow_up}?"
                else:
                    follow_up = f"Well, {follow_up}?"
            
            # Store the follow-up question in conversation memory
            self.conversation_memory.append({
                "question": follow_up,
                "follow_up": True,
                "timestamp": datetime.now().isoformat()
            })
            
            return follow_up
            
        except Exception as e:
            if "exceed context window" in str(e):
                # Clear caches and return a simpler follow-up
                self.clear_all_caches()
                return "What would you like to know more about?"
            else:
                # For other errors, clear caches and return a fallback
                self.clear_all_caches()
                return "Would you like to explore another aspect of this topic?"

    def answer_followup_node(self, state: State) -> dict:
        """
        Handle follow-up questions by answering based on previous conversation context.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            dict: Updated state with the follow-up answer
        """
        messages = state["messages"]
        input_text = state["input"]
        
        response = self._answer_follow_up(input_text)
        messages.append(AIMessage(content=response))
        
        return {"messages": messages}
        
    def _determine_next_node(self, state: State) -> str:
        """
        Determine whether to route to retrieval or answer_followup based on the input.
        Uses LLM to make a more nuanced decision about whether the input is a follow-up question.
        
        Args:
            state (State): Current conversation state
            
        Returns:
            str: Either "retrieval" or "answer_followup" based on the analysis
        """
        # Get the user's input
        user_input = state.get("input", "").lower()
        
        # If there's no conversation history, it's not a follow-up
        if not self.conversation_memory:
            return "retrieval"
            
        # Get the last non-follow-up exchange for context
        last_exchange = next(
            (msg for msg in reversed(self.conversation_memory)
            if not msg.get("follow_up", False)),
            None
        )
        
        if not last_exchange:
            return "retrieval"
            
        # Create a prompt for the LLM to analyze if this is a follow-up question
        prompt = f"""Analyze if the following user input is a follow-up question to the previous conversation.
Consider the context and determine if the user is asking for clarification or additional information about the previous answer.

Previous Answer:
{last_exchange.get('answer', '')}

Current User Input:
{user_input}

Guidelines for determining if it's a follow-up:
1. Is the user asking for clarification about something mentioned in the previous answer?
2. Is the user asking for more details about a specific point from the previous answer?
3. Is the user using phrases like "I don't know", "you tell me", or "tell me"?
4. Is the user asking about a specific aspect mentioned in the previous answer?
5. Is the question directly related to the previous discussion?

Respond with only one word: "followup" if it's a follow-up question, or "retrieval" if it's a new question."""

        # Get the LLM's decision
        response = self.llm.query(prompt)
        decision = response.content.strip().lower()
        
        return "answer_followup" if decision == "followup" else "retrieval"

    def _generate_clarification(self, state: State) -> str:
        """
        Generate a clarification message when uncertainty is detected in the content.
        
        Args:
            state (State): Current conversation state containing messages and context
            
        Returns:
            str: A clarification message to help resolve uncertainty
        """
        # Get the retrieval message to analyze the content
        retrieval_msg = next(
            (msg for msg in state["messages"]
            if isinstance(msg, ToolMessage) and msg.name == "document_retriever"),
            None
        )
        
        if not retrieval_msg:
            return "I apologize, but I'm having trouble understanding the context. Could you please rephrase your question?"
            
        # Extract the content and user's question
        content = retrieval_msg.content.replace("Retrieved context:\n", "").strip()
        user_question = state.get("input", "")
        
        # Create a prompt for the LLM to identify the specific areas of uncertainty
        prompt = f"""Analyze the following content and question to identify areas of uncertainty and generate a helpful clarification request.

Content:
{content}

User Question:
{user_question}

Guidelines for generating clarification:
1. Identify specific parts of the content that are unclear or ambiguous
2. Point out any missing or incomplete information
3. Ask for clarification in a friendly, conversational tone
4. Focus on the most important aspects that need clarification
5. Keep the clarification request concise and specific
6. Use natural language that a human would use
7. Avoid technical jargon unless necessary
8. Make sure the clarification request is directly related to the user's question
9. If multiple aspects need clarification, prioritize the most important ones
10. End with an open-ended question to encourage user engagement

Generate a natural clarification request:"""
        
        # Get the clarification from the LLM
        response = self.llm.query(prompt)
        clarification = response.content if isinstance(response, AIMessage) else str(response)
        
        # Clean up the response
        clarification = clean_output(clarification)
        
        # Add a friendly prefix if not already present
        if not any(clarification.lower().startswith(phrase) for phrase in [
            "i'm not sure", "i'm unclear", "could you clarify",
            "i need more information", "i'm having trouble understanding"
        ]):
            clarification = f"I'm not entirely sure about this, but {clarification}"
            
        return clarification

if __name__ == "__main__":
    # Clear cache if requested
    if "--clear-cache" in sys.argv:
        print("Clearing cache...")
        if os.path.exists(config["HF_HOME"]):
            shutil.rmtree(config["HF_HOME"])
            os.makedirs(config["HF_HOME"])
        print("Cache cleared.")
    
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
