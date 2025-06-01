from typing import List, Dict, Any
import json
from pathlib import Path
import re

class TextProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text processor with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_json_articles(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and parse JSON articles."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_text_file(self, file_path: str) -> str:
        """Load text from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to find a good breaking point
            if end < len(text):
                # Look for the last period or newline within the last 100 characters
                break_point = text.rfind('.', start, end)
                if break_point == -1:
                    break_point = text.rfind('\n', start, end)
                if break_point != -1:
                    end = break_point + 1
            
            # Add the chunk
            chunks.append(text[start:end].strip())
            
            # Move the start pointer, accounting for overlap
            start = end - self.chunk_overlap
        
        return chunks

    def process_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single article into chunks with metadata.
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            List of chunks with metadata
        """
        # Combine relevant article fields
        content = f"Title: {article.get('content_headline', '')}\n"
        content += f"Content: {article.get('body', '')}\n"
        content += f"Categories: {', '.join(cat.get('content_category', '') for cat in article.get('content_categories', []))}\n"
        content += f"Tags: {', '.join(tag.get('name', '') for tag in article.get('content_tags', []))}"
        
        # Chunk the content
        chunks = self.chunk_text(content)
        
        # Add metadata to each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'article_id': article.get('content_id', ''),
                'chunk_id': i,
                'type': 'article'
            })
        
        return processed_chunks

    def process_policy_document(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Process a policy document into chunks with metadata.
        
        Args:
            text: Policy document text
            source: Source identifier for the policy document
            
        Returns:
            List of chunks with metadata
        """
        chunks = self.chunk_text(text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'source': source,
                'chunk_id': i,
                'type': 'policy'
            })
        
        return processed_chunks 