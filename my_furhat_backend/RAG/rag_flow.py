import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import LlamaCppEmbeddings

# Set up logging configuration for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    """
    A class to manage Retrieval-Augmented Generation (RAG) tasks using a Chroma vector store.

    This class supports:
      - Loading and chunking documents (from PDFs),
      - Creating or loading a persistent Chroma vector store,
      - Populating the vector store with document chunks,
      - Retrieving similar document chunks with an option to rerank results using a cross-encoder.
    """

    def __init__(self, hf=True, **kwargs):
        """
        Initialize the RAG instance with either HuggingFace or LlamaCpp embeddings.

        Parameters:
            hf (bool): If True, use HuggingFaceEmbeddings; if False, use LlamaCppEmbeddings.
            **kwargs: Additional configuration options, including:
                - model_name_hf: Name of the HuggingFace model (default: "sentence-transformers/all-MiniLM-L6-v2").
                - model_path_llama: Path to the LlamaCpp model (default provided).
                - persist_directory: Directory path for persisting the vector store (default: "my_furhat_backend/db").
                - path_to_document: Path to the document file to load (default: "my_furhat_backend/rag_document.txt").
        """
        if hf: 
            # Initialize embeddings using a HuggingFace model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=kwargs.get("model_name_hf", "sentence-transformers/all-MiniLM-L6-v2")
            )
        else:
            # Initialize embeddings using a LlamaCpp model
            self.embeddings = LlamaCppEmbeddings(
                model_path=kwargs.get("model_path_llama", "my_furhat_backend/ggufs_models/all-MiniLM-L6-v2-Q4_K_M.gguf")
            )
        # Set the directory to persist the vector store
        self.persist_directory = kwargs.get("persist_directory", "my_furhat_backend/db")
        # Set the document path to be loaded
        self.path_to_docs = kwargs.get("path_to_document", "my_furhat_backend/rag_document.txt")
        # Create and populate the Chroma vector store (either load existing or create new)
        self.vector_store = self.__create_and_populate_chroma()
        
    def __load_docs(self):
        """
        Load documents from a PDF file using PyPDFLoader.

        Returns:
            List[Document]: A list of Document objects loaded from the file.
        """
        try:
            # Instantiate a loader with the provided document path
            loader = PyPDFLoader(self.path_to_docs)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} document(s) from {self.path_to_docs}.")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents from {self.path_to_docs}: {e}")
            return []
    
    def __load_and_chunk_docs(self):
        """
        Load documents and split them into smaller text chunks using RecursiveCharacterTextSplitter.

        Returns:
            List[Document]: A list of document chunks derived from the loaded documents.
        """
        docs = self.__load_docs()
        if not docs:
            logger.error("No documents loaded; cannot perform chunking.")
            return []
        # Split documents into chunks with specified size and overlap for better context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(chunks)} chunk(s).")
        return chunks
    
    def __populate_chroma(self, vector_store):
        """
        Populate the given Chroma vector store with document chunks.

        Parameters:
            vector_store (Chroma): The Chroma vector store instance to populate.
        """
        # Load and split the documents into chunks
        chunks = self.__load_and_chunk_docs()
        if chunks:
            # Add the chunks to the vector store
            _ = vector_store.add_documents(documents=chunks)
            logger.info("Populated vector store with document chunks.")
        else:
            logger.warning("No document chunks available to populate the vector store.")
        
    def __create_and_populate_chroma(self):
        """
        Create a new Chroma vector store or load an existing one, and populate it with documents.

        Returns:
            Chroma: The created or loaded and populated vector store.
        """
        if os.path.exists(self.persist_directory):
            # If the persist directory exists, load the existing vector store
            logger.info("Persist directory exists. Loading existing vector store.")
            return self.__load_db()
        else:
            # Otherwise, create a new vector store from the document chunks
            logger.info("Creating new Chroma vector store.")
            vector_store = Chroma.from_documents(self.__load_and_chunk_docs(), self.embeddings, persist_directory=self.persist_directory)
            return vector_store
    
    def __load_db(self):
        """
        Load an existing Chroma vector store from the persist directory.

        Returns:
            Chroma: The loaded vector store.
        """
        logger.info(f"Loading vector store from {self.persist_directory}.")
        return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
    
    def add_docs_to_db(self, path_to_docs):
        """
        Add new documents to the existing vector store.

        Parameters:
            path_to_docs (str): The file path to the new document(s) that should be added.
        """
        # Update the document path to the new documents
        self.path_to_docs = path_to_docs
        # Populate the current vector store with the new documents
        self.__populate_chroma(self.vector_store)
    
    def retrieve_similar(self, query_text, top_n=5, search_kwargs=20, rerank=True):
        """
        Retrieve document chunks similar to the given query text.

        Parameters:
            query_text (str): The query string used to search the vector store.
            top_n (int): The number of top documents to return after reranking.
            search_kwargs (int): The number of documents to retrieve initially.
            rerank (bool): Whether to rerank the initial retrieval using a cross-encoder.

        Returns:
            List[Document]: A list of relevant document chunks matching the query.
        """
        if rerank:
            logger.info("Reranking documents...")
            return self.__rerank(query_text, top_n=top_n, search_kwargs=search_kwargs)
        # If no reranking is needed, perform a direct similarity search on the vector store
        return self.vector_store.similarity_search(query_text)
    
    def get_document_context(self, document: str) -> str:
        """
        Retrieve the context of a document from the DocumentAgent.

        Generates a prompt instructing the retrieval system to extract the main context and themes from the document.
        The function then queries the retrieval system (rag_instance) to obtain the document context.

        Parameters:
            document (str): The name or identifier of the document to retrieve context for.

        Returns:
            str: The context of the specified document.
        """
        # Construct a prompt that asks for the document's overarching context and main themes.
        prompt = (
            "Extract the overarching context and main themes from the document titled "
            f"'{document}'. Focus on summarizing the key topics, narrative, and any major findings "
            "without including extraneous details."
        )
        # Retrieve and return the document context using the retrieval instance.
        return self.retrieve_similar(prompt)
    
    def __rerank(self, prompt, top_n=5, search_kwargs=20):
        """
        Rerank retrieved documents using a cross-encoder for improved relevance.

        Parameters:
            prompt (str): The query or prompt text to compare against document chunks.
            top_n (int): The number of top documents to return after reranking.
            search_kwargs (int): The number of documents to retrieve initially for reranking.

        Returns:
            List[Document]: A list of document chunks reordered by relevance.
        """
        # Initialize a cross-encoder model for reranking
        model = HuggingFaceCrossEncoder(model_name="mixedbread-ai/mxbai-rerank-base-v1")
        # Set up the reranker using the cross-encoder model
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        # Create a contextual compression retriever that uses the vector store as the base retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": search_kwargs})
        )
        # Retrieve and rerank documents based on the provided prompt
        compressed_docs = compression_retriever.invoke(prompt)
        return compressed_docs
