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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    """
    A class to manage Retrieval-Augmented Generation (RAG) tasks using a Chroma vector store.
    
    It supports loading and chunking documents from a PDF, populating or loading a persistent
    Chroma vector store, and retrieving similar document chunks—with optional reranking.
    """
    def __init__(self, hf=True, **kwargs):
        """
        Initialize the RAG instance with either HuggingFace or LlamaCpp embeddings.
        
        Parameters:
            hf (bool): If True, use HuggingFaceEmbeddings; otherwise, use LlamaCppEmbeddings.
            **kwargs: Additional configuration options, including:
                - model_name_hf: Name of the HuggingFace model.
                - model_path_llama: Path to the LlamaCpp model.
                - persist_directory: Directory for persisting the vector store.
                - path_to_document: Path to the document file.
        """
        if hf: 
            self.embeddings = HuggingFaceEmbeddings(
                model_name=kwargs.get("model_name_hf", "sentence-transformers/all-MiniLM-L6-v2")
            )
        else:
            self.embeddings = LlamaCppEmbeddings(
                model_path=kwargs.get("model_path_llama", "my_furhat_backend/ggufs_models/all-MiniLM-L6-v2-Q4_K_M.gguf")
            )
        self.persist_directory = kwargs.get("persist_directory", "my_furhat_backend/db")
        self.path_to_docs = kwargs.get("path_to_document", "my_furhat_backend/rag_document.txt")
        self.vector_store = self.__create_and_populate_chroma()
        
    
    def __load_docs(self):
        """
        Load documents using PyPDFLoader.
        
        Returns:
            List[Document]: A list of loaded Document objects.
        """
        try:
            loader = PyPDFLoader(self.path_to_docs)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} document(s) from {self.path_to_docs}.")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents from {self.path_to_docs}: {e}")
            return []
    
    def __load_and_chunk_docs(self):
        """
        Load documents and split them into smaller chunks using RecursiveCharacterTextSplitter.
        
        Returns:
            List[Document]: A list of document chunks.
        """
        docs = self.__load_docs()
        if not docs:
            logger.error("No documents loaded; cannot perform chunking.")
            return []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Split documents into {len(chunks)} chunk(s).")
        return chunks
    
    def __populate_chroma(self, vector_store):
        """
        Populate the vector store with document chunks.
        
        Parameters:
            vector_store: The Chroma vector store instance to populate.
        """
        chunks = self.__load_and_chunk_docs()
        if chunks:
            _ = vector_store.add_documents(documents=chunks)
            logger.info("Populated vector store with document chunks.")
        else:
            logger.warning("No document chunks available to populate the vector store.")
        
    def __create_and_populate_chroma(self):
        """
        Create a new Chroma vector store and populate it with documents.
        
        Returns:
            Chroma: The created and populated vector store.
        """
        if os.path.exists(self.persist_directory):
            logger.info("Persist directory exists. Loading existing vector store.")
            return self.__load_db()
        else:
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
            path_to_docs (str): The path to the new document(s) to be added.
        """
        self.path_to_docs = path_to_docs
        self.__populate_chroma(self.vector_store)
    
    def retrieve_similar(self, query_text, top_n=5, search_kwargs=10, rerank=True):
        """
        Retrieve similar document chunks to the given query.
        
        Parameters:
            query_text (str): The query string.
            top_n (int): Number of top documents to return after reranking.
            search_kwargs (int): Number of documents to retrieve initially.
            rerank (bool): If True, perform reranking on the retrieved results.
            
        Returns:
            List[Document]: A list of relevant document chunks.
        """
        if rerank:
            logger.info("Reranking documents...")
            return self.__rerank(query_text, top_n=top_n, search_kwargs=search_kwargs)
        return self.vector_store.similarity_search(query_text)
    
    def __rerank(self, prompt, top_n=5, search_kwargs=20):
        """
        Rerank retrieved documents using a cross-encoder.
        
        Parameters:
            prompt (str): The query or prompt text.
            top_n (int): Number of top documents to return after reranking.
            search_kwargs (int): Number of documents to retrieve initially.
            
        Returns:
            List[Document]: A list of reranked document chunks.
        """
        model = HuggingFaceCrossEncoder(model_name="mixedbread-ai/mxbai-rerank-base-v1")
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": search_kwargs})
        )
        compressed_docs = compression_retriever.invoke(prompt)
        return compressed_docs
