from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_model = None

def get_model():
    global _model
    try:
        if _model is None:
            logger.info("Loading sentence transformer model...")
            _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        return _model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

class VectorStore:
    def __init__(self):
        """Initialize the vector store."""
        try:
            logger.info("Initializing VectorStore...")
            self.model = get_model()
            self.index = None
            self.documents = []
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Vector dimension: {self.dimension}")
            logger.info("VectorStore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and metadata
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return

            logger.info(f"Adding {len(documents)} documents to vector store...")
            
            # Extract texts and store documents
            texts = [doc['text'] for doc in documents]
            self.documents.extend(documents)
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Initialize or update FAISS index
            if self.index is None:
                logger.info("Creating new FAISS index...")
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Add vectors to the index
            logger.info("Adding vectors to FAISS index...")
            self.index.add(embeddings.astype('float32'))
            logger.info("Documents added successfully")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        try:
            if self.index is None or len(self.documents) == 0:
                logger.warning("No documents in vector store")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search the index
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Prepare results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):  # Ensure index is valid
                    # Convert distance to similarity score (1 / (1 + distance))
                    similarity = 1 / (1 + distance)
                    results.append((self.documents[idx], similarity))
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    def get_article_by_id(self, article_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific article by its ID.
        
        Args:
            article_id: ID of the article to retrieve
            
        Returns:
            Article document if found, None otherwise
        """
        try:
            for doc in self.documents:
                if doc.get('type') == 'article' and doc.get('article_id') == article_id:
                    return doc
            return None
        except Exception as e:
            logger.error(f"Error retrieving article: {str(e)}")
            raise

    def clear(self) -> None:
        """Clear all documents and reset the index."""
        try:
            self.documents = []
            self.index = None 
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise 