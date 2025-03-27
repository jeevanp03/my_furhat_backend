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
from typing import Dict, Tuple, Optional
from datetime import datetime
from transformers import pipeline
import re

# Import custom LLM/chatbot classes.
from my_furhat_backend.models.chatbot_factory import create_chatbot
from my_furhat_backend.utils.util import clean_output, summarize_text
# Import the RAG (Retrieval-Augmented Generation) class.
from my_furhat_backend.RAG.rag_flow import RAG

context = None
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
    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the DocumentAgent.

        Sets up the persistent memory for checkpointing and builds the state graph defining the conversation flow.
        """
        # Initialize MemorySaver to checkpoint the state graph.
        self.memory = MemorySaver()
        
        # Initialize RAG with caching
        self.rag_instance = RAG(
            hf=True,
            persist_directory="my_furhat_backend/db",
            path_to_document="my_furhat_backend/ingestion/NorwAi annual report 2023.pdf"
        )
        
        # Initialize chatbot with optimized settings
        self.chatbot = create_chatbot(
            "llama",
            model_id=model_id,
            n_ctx=4096,  # Reduced context window
            n_batch=512,  # Increased batch size
            n_threads=4,  # Optimize thread count
            n_gpu_layers=32  # Use more GPU layers
        )
        self.llm = self.chatbot.llm
        
        # Create a state graph using the custom State schema.
        self.graph = StateGraph(State)
        
        # Initialize caches with larger sizes
        self.question_cache = QuestionCache()
        self.context_cache = {}
        self.summary_cache = {}  # New cache for document summaries
        
        # Build the conversation flow by adding nodes and edges.
        self._build_graph()
        
        # Compile the graph with checkpointing support.
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        # Initialize conversation memory with a larger size
        self.conversation_memory = []
        self.max_memory_size = 10  # Increased from 5
        
        # Initialize sentiment analyzer with specific model and caching
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
          - Response formatting.
          - Follow-up question handling.

        It also sets up edges for linear progression and conditional branching based on LLM decisions.
        """
        # Add nodes to the graph with corresponding callback functions.
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
        retrieved_docs = self.rag_instance.retrieve_similar(query, rerank=True)
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
        Implements caching to avoid re-summarizing the same content.

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
            
            # Check if we have a cached summary for this content
            content_hash = hash(text_to_summarize)
            if content_hash in self.summary_cache:
                summarized_text = self.summary_cache[content_hash]
            else:
                # Generate a summary of the retrieved content with appropriate parameters
                summarized_text = summarize_text(
                    text_to_summarize,
                    max_length=400,
                    min_length=50
                )
                # Cache the summary
                self.summary_cache[content_hash] = summarized_text
                
                # Limit cache size to prevent memory issues
                if len(self.summary_cache) > 1000:
                    # Remove oldest entry
                    self.summary_cache.pop(next(iter(self.summary_cache)))
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
        3. Overall quality and completeness

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
        # Delegate response generation to the chatbot's conversation method
        return self.chatbot.chatbot(state)
    
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
4. Use contractions and casual language where appropriate
5. Add conversational transitions between topics
6. Avoid meta-commentary about being human or conversational
7. Keep responses concise and focused

Respond with just the reformatted text."""

        # Get the LLM's formatted response
        response = self.llm.query(prompt)
        formatted_content = response.content.strip()
        
        # Replace the last AI message with the formatted version
        state["messages"][-1] = AIMessage(content=formatted_content)
        
        return state
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze the sentiment of the text and return a score between -1 and 1."""
        result = self.sentiment_analyzer(text)[0]
        return 1.0 if result['label'] == 'POSITIVE' else -1.0
        
    def _adjust_tone(self, text: str, sentiment: float) -> str:
        """Adjust the tone of the response based on sentiment and personality."""
        # Add personality-based modifiers
        if self.personality_traits["enthusiasm"] > 0.5:
            text = text.replace(".", "!")
            
        # Adjust based on sentiment
        if sentiment < -0.5:
            text = f"I understand this might be concerning. {text}"
        elif sentiment > 0.5:
            text = f"That's great to hear! {text}"
            
        return text
        
    def _generate_engaging_prompt(self, document_name: str, answer: str) -> str:
        """Generate an engaging follow-up prompt based on conversation history and personality."""
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
            
    def _update_conversation_memory(self, question: str, answer: str):
        """Update the conversation memory with new Q&A pair."""
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
        """Get a summary of the conversation context."""
        if not self.conversation_memory:
            return ""
            
        context = "Previous conversation:\n"
        for exchange in self.conversation_memory:
            context += f"Q: {exchange['question']}\nA: {exchange['answer']}\n"
        return context
        
    def run(self, initial_input: str, system_prompt: str = None) -> str:
        """
        Execute the document agent's conversation flow.

        Parameters:
            initial_input (str): The user's query to initiate the conversation.
            system_prompt (str, optional): An optional system prompt to set the conversational context.

        Returns:
            str: The cleaned output from the final AI-generated message.
        """
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
            # Store in conversation memory with document name
            self.conversation_memory.append({
                "question": initial_input,
                "answer": final_ai_msg.content,
                "document_name": "CMRPublished",  # Default document name
                "timestamp": datetime.now().isoformat()
            })
            return clean_output(final_ai_msg.content)
        
        return "No response generated."

    def _answer_follow_up(self, follow_up_question: str) -> str:
        """
        Answer a follow-up question directly without treating it as a new query.
        
        Parameters:
            follow_up_question (str): The follow-up question to answer.
            
        Returns:
            str: The answer to the follow-up question.
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
        
        Parameters:
            document_name (str): Name of the document being discussed.
            answer (str): The previous answer to generate a follow-up for.
            
        Returns:
            str: A conversational follow-up question.
        """
        # Get document context from cache or retrieve it
        if document_name not in self.context_cache:
            self.context_cache[document_name] = self.rag_instance.get_document_context(document_name)
        
        # Extract text content from document list
        context_docs = self.context_cache[document_name]
        context = "\n".join(doc.page_content for doc in context_docs)
        
        # Truncate context and answer to prevent token overflow
        max_context_length = 300  # Increased from 200 to 300
        max_answer_length = 400   # Increased from 300 to 400

        summarized_context = summarize_text(context, max_length=400, min_length=150)  # Increased limits
        summarized_answer = summarize_text(answer, max_length=400, min_length=150)    # Increased limits
        
        context_words = summarized_context.split()
        answer_words = summarized_answer.split()
        
        if len(context_words) > max_context_length:
            context = ' '.join(context_words[:max_context_length]) + '...'
        if len(answer_words) > max_answer_length:
            answer = ' '.join(answer_words[:max_answer_length]) + '...'
        
        # Create a prompt for the LLM to generate a natural follow-up
        prompt = f"""Based on the previous answer, generate a single, natural follow-up question.
The question should be directly related to what was just discussed.

Previous Answer:
{answer}

Guidelines:
1. Make it sound like a natural question someone would ask in conversation
2. Use casual, everyday language
3. Keep it short and simple
4. Focus on an interesting aspect from the previous answer
5. Avoid formal or academic language
6. Don't use phrases like "as discussed" or "within this context"
7. Don't ask about the user's experience or opinions
8. Don't use complex terminology unless it's essential to the topic
9. Make sure the question is directly related to the previous answer
10. Don't introduce completely new topics

Generate a single, natural follow-up question:"""
        
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
        
        # Add a conversational prefix
        follow_up = f"I'm curious, {follow_up}?"
        
        # Store the follow-up question in conversation memory
        self.conversation_memory.append({
            "question": follow_up,
            "follow_up": True,
            "timestamp": datetime.now().isoformat()
        })
        
        return follow_up

    def answer_followup_node(self, state: State) -> dict:
        """
        Handle follow-up questions by answering based on previous conversation context.

        Parameters:
            state (State): The current conversation state.

        Returns:
            dict: Updated state with the follow-up answer.
        """
        # Get the last follow-up question from the conversation
        last_follow_up = next(
            (msg for msg in reversed(self.conversation_memory)
            if "follow_up" in msg),
            None
        )
        
        if not last_follow_up:
            state["messages"].append(AIMessage(
                content="I apologize, but I don't have enough context to provide a meaningful answer."
            ))
            return {"messages": state["messages"]}
            
        # Get the previous answer from conversation memory
        previous_exchange = next(
            (msg for msg in reversed(self.conversation_memory)
            if not msg.get("follow_up", False)),  # Get the last non-follow-up exchange
            None
        )
        
        if not previous_exchange:
            state["messages"].append(AIMessage(
                content="I apologize, but I don't have enough context to provide a meaningful answer."
            ))
            return {"messages": state["messages"]}
            
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
{last_follow_up["question"]}

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
        
        # Add the answer to the state messages
        state["messages"].append(AIMessage(content=clean_output(answer)))
        
        return {"messages": state["messages"]}

    def _determine_next_node(self, state: State) -> str:
        """
        Determine whether to route to retrieval or answer_followup based on the input.
        Uses LLM to make a more nuanced decision about whether the input is a follow-up question.

        Parameters:
            state (State): The current conversation state.

        Returns:
            str: Either "retrieval" or "answer_followup" based on the analysis.
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
        print("Agent Follow-up:", agent.engage("CMRPublished", response))
