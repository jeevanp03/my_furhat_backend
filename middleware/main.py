"""
This module implements a FastAPI server that provides endpoints for interacting with the DocumentAgent.
The server exposes four endpoints:

1. /ask (POST):
   - Receives a transcription (user's spoken text) and processes it synchronously through the DocumentAgent.
   - Returns the generated answer as a JSON response.

2. /transcribe (POST):
   - Receives a transcription from the Furhat frontend.
   - Processes the transcription asynchronously by passing it to the DocumentAgent.
   - Stores the agent's response for later retrieval.
   - Returns an acknowledgment that the transcription was received.

3. /response (GET):
   - Retrieves the latest response generated by the DocumentAgent.
   - Returns the response in a JSON format so that the Furhat frontend can vocalize it.

4. /get_docs (POST):
   - Retrieves a list of documents via get_list_docs().
   - If one document is found, returns it.
   - If no documents are found, returns an appropriate message.
   - If multiple documents are found, uses TextClassifier to rank them based on the transcription
     and returns the top-ranked document.

5. /engage (POST):
   - Retrieves document context, creates a combined text with the provided answer, generates a summary,
     and then creates a followup prompt.

The server integrates with a backend agent (DocumentAgent) which is responsible for managing the conversation flow,
retrieving relevant document context, and generating responses using an LLM.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio

# Import the DocumentAgent and utility functions from the backend.
from my_furhat_backend.agents.document_agent import DocumentAgent
from my_furhat_backend.utils.util import (
    get_list_docs, 
    summarize_text, 
    get_document_context, 
    generate_followup_prompt
)
from my_furhat_backend.models.classifier import TextClassifier

# Initialize the FastAPI application.
app = FastAPI()


class Transcription(BaseModel):
    """
    Pydantic model representing the transcription received from the client.
    
    Attributes:
        content (str): The text content of the transcription.
    """
    content: str


class EngageRequest(BaseModel):
    """
    Pydantic model for the engage endpoint containing the document and the answer.
    
    Attributes:
        document (str): Identifier or content of the document to be used.
        answer (str): The answer generated or provided that will be used to create further prompts.
    """
    document: str
    answer: str


# For demonstration purposes, using a simple in-memory store for the latest response.
# In production, consider using a more robust solution (e.g., session management or a database).
latest_response = None
context = None

# Instantiate the DocumentAgent to handle LLM processing.
agent = DocumentAgent()


@app.post("/ask", response_model=dict)
async def ask_question(transcription: Transcription):
    """
    Process a transcription by running it through the LLM agent synchronously and return the generated response.

    This endpoint receives a POST request containing a transcription (user's spoken text),
    passes the transcription to the DocumentAgent's `run` method to generate an answer, and returns
    the result in a JSON response. The agent processing is offloaded to a separate thread to avoid blocking.
    If an error occurs during processing, a 500 HTTPException is raised.

    Parameters:
        transcription (Transcription): A Pydantic model instance containing the transcribed text.

    Returns:
        dict: A JSON response with a 'response' key that holds the generated answer as a string.
    """
    global latest_response
    try:
        # Offload the agent processing to a separate thread to avoid blocking the event loop.
        latest_response = await asyncio.to_thread(agent.run, transcription.content)
    except Exception as e:
        # If any error occurs, raise a HTTP 500 error with the error message.
        raise HTTPException(status_code=500, detail=str(e))
    return {"response": latest_response}


@app.post("/transcribe", response_model=dict)
async def transcribe(transcription: Transcription):
    """
    Asynchronously process transcribed text from the Furhat frontend and store the agent's response.

    This endpoint accepts a POST request containing the transcription (user's spoken text),
    processes it using the DocumentAgent's `run` method (offloaded to a thread), and stores the generated response
    in a global variable for later retrieval via the /response endpoint. If an error occurs during processing,
    a 500 HTTPException is raised.

    Parameters:
        transcription (Transcription): A Pydantic model instance with the transcribed text.

    Returns:
        dict: A JSON response indicating that the transcription was received.
    """
    global latest_response
    try:
        # Process the transcription asynchronously and update the global latest_response.
        latest_response = await asyncio.to_thread(agent.run, transcription.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Acknowledge that the transcription has been received.
    return {"status": "transcription received"}


@app.get("/response", response_model=dict)
async def get_response():
    """
    Retrieve the latest response generated by the LLM agent.

    This endpoint accepts a GET request and returns the most recent response stored from a transcription.
    If no response is available, it returns a message indicating that no response has been generated yet.

    Returns:
        dict: A JSON response with a 'response' key containing the agent's answer as a string.
    """
    if latest_response is None:
        # Return a default message if no response has been generated yet.
        return {"response": "No response generated yet."}
    return {"response": latest_response}


@app.post("/get_docs", response_model=dict)
async def get_docs(transcription: Transcription):
    """
    Retrieve a document based on the provided transcription by using document retrieval and classification.

    This endpoint obtains a list of documents using `get_list_docs()`. If no documents are found,
    it returns a message indicating such. If exactly one document is found, it returns that document.
    If multiple documents are found, it uses a TextClassifier to rank the documents based on the transcription
    and returns the top-ranked document.

    Parameters:
        transcription (Transcription): A Pydantic model instance containing the transcribed text
                                         to be used for ranking the documents.

    Returns:
        dict: A JSON response with a 'response' key that contains the relevant document or an appropriate message.
    """
    try:
        # Retrieve a list of available documents.
        docs = await asyncio.to_thread(get_list_docs)
        if not docs:
            return {"response": "No documents found."}
        if len(docs) == 1:
            # If only one document exists, return it directly.
            return {"response": docs[0]}
        # If multiple documents are found, rank them using the text classifier.
        ranked_docs = await asyncio.to_thread(TextClassifier().classify, transcription.content, docs)
        # Assuming ranked_docs is a dictionary with documents as keys and scores as values,
        # select the top-ranked document (first key in the dictionary).
        top_doc = list(ranked_docs.keys())[0] if ranked_docs else "No document ranked."
        return {"response": top_doc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engage", response_model=dict)
async def engage(engage_request: EngageRequest):
    """
    Process an engagement request by retrieving document context, generating a summary, and producing a followup prompt.

    This endpoint retrieves additional context for the specified document, combines it with the provided answer,
    generates a summary of this combined text, and then creates a followup prompt for further engagement.
    If any error occurs during this processing chain, a 500 HTTPException is raised.

    Parameters:
        engage_request (EngageRequest): A Pydantic model instance containing the document identifier and answer.

    Returns:
        dict: A JSON response with a 'prompt' key that contains the generated followup prompt.
    """
    try:
        # Retrieve context for the given document.
        global context
        if not context:
            context = await asyncio.to_thread(get_document_context, engage_request.document)
        # Combine the answer with the retrieved context.
        combined_text = f"{engage_request.answer}\nContext: {context}"
        # Generate a summary of the combined text.
        summary = await asyncio.to_thread(summarize_text, combined_text)
        # Generate a followup prompt based on the summary.
        prompt = await asyncio.to_thread(generate_followup_prompt, summary)
        return {"prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application using Uvicorn if this module is executed directly.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
