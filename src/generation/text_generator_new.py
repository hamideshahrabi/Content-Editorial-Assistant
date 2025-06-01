from typing import List, Dict, Any, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model=None, tokenizer=None):
        """Initialize the text generator with optional pre-loaded model and tokenizer."""
        self.device = "cpu"  # Force CPU usage
        self.model = model
        self.tokenizer = tokenizer
        
        if self.model is None or self.tokenizer is None:
            logger.info("Loading Flan-T5 model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                'google/flan-t5-large',
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                'google/flan-t5-large',
                local_files_only=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            logger.info("Model and tokenizer loaded successfully")

    def _generate(self, prompt: str, max_length: int = 512, min_length: int = None) -> str:
        """Generate text based on the prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            generation_kwargs = {
                "max_length": max_length,
                "num_beams": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "no_repeat_ngram_size": 3
            }
            if min_length is not None:
                generation_kwargs["min_length"] = min_length
            outputs = self.model.generate(**inputs, **generation_kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    def generate_headline(self, content: str) -> str:
        """Generate an SEO-optimized headline."""
        prompt = f"""Write a compelling SEO-optimized headline for this article. The headline must:
1. Be a complete, grammatically correct sentence
2. Include the most important keywords at the beginning
3. Be clear, descriptive, and informative
4. Avoid clickbait or sensationalism
5. Be between 50-100 characters long
6. Capture the main story and its significance
7. Use active voice and present tense
8. Include specific details when relevant
9. Be engaging and newsworthy
10. Include numbers or statistics when available

Article Content: {content[:500]}

SEO Headline:"""
        
        headline = self._generate(prompt, max_length=100, min_length=50)
        headline = headline.strip().strip('"\'')
        
        # Ensure headline is a complete sentence and within length limits
        if len(headline) > 100:
            match = re.match(r'(.{20,100}[.!?])', headline)
            if match:
                headline = match.group(1)
            else:
                last_space = headline[:100].rfind(' ')
                headline = headline[:last_space] if last_space > 0 else headline[:100]
        
        return headline

    def generate_social_headline(self, content: str) -> str:
        """Generate a social media-optimized headline."""
        prompt = f"""Write an engaging social media headline for this article. The headline must:
1. Be attention-grabbing and shareable
2. Use conversational language
3. Include relevant hashtags
4. Be under 100 characters
5. Create curiosity and encourage clicks
6. Be authentic and avoid clickbait
7. Include numbers or statistics when available

Article Content: {content[:500]}

Social Media Headline:"""
        
        headline = self._generate(prompt, max_length=100)
        return headline.strip().strip('"\'')[:100]

    def generate_summary(self, content: str) -> str:
        """Generate a concise summary of the content."""
        prompt = f"""Write a concise summary of this article. The summary must:
1. Capture the main points
2. Be clear and informative
3. Be under 280 characters
4. Use active voice
5. Include key statistics or numbers
6. Be engaging and shareable

Article Content: {content[:500]}

Summary:"""
        
        summary = self._generate(prompt, max_length=280)
        return summary.strip().strip('"\'')[:280]

    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer based on the context and question."""
        prompt = f"""Based on the following context, answer this question: {question}

Context:
{context}

Answer:"""
        
        answer = self._generate(prompt, max_length=512)
        return answer.strip().strip('"\'') 