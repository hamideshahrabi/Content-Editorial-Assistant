from typing import List, Dict, Any, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
# Use the smallest available sentence-transformer for embeddings to minimize memory usage
# all-MiniLM-L6-v2 is one of the smallest and most efficient for this purpose
_model = None
_tokenizer = None

def get_device():
    """Get the best available device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_model_and_tokenizer():
    global _model, _tokenizer
    try:
        if _model is None or _tokenizer is None:
            logger.info("Loading Flan-T5 model and tokenizer...")
            device = get_device()
            logger.info(f"Using device: {device}")
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using cache directory: {cache_dir}")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(
                'google/flan-t5-large',
                cache_dir=cache_dir
            )
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            logger.info("Loading model...")
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                'google/flan-t5-large',
                cache_dir=cache_dir,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
            )
            if device in ["cuda", "mps"]:
                _model = _model.to(device)
                logger.info(f"Model moved to {device}")
            elif device == "cpu":
                logger.info("Model using CPU")
            logger.info("Model loaded successfully")
        return _model, _tokenizer
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

class TextGenerator:
    def __init__(self):
        """Initialize the text generator."""
        try:
            logger.info("Initializing TextGenerator...")
            self.device = get_device()
            logger.info(f"Using device: {self.device}")
            self.model, self.tokenizer = get_model_and_tokenizer()
            logger.info("TextGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TextGenerator: {str(e)}")
            raise

    def _generate(self, prompt: str, max_length: int = 512, min_length: int = None) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: Input prompt for generation
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text (optional)
            
        Returns:
            Generated text
        """
        try:
            logger.info(f"Generating text with max_length={max_length}...")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
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
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Text generation completed successfully")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    def answer_policy_question(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer to a policy question based on retrieved context.
        
        Args:
            question: The policy question
            context: List of relevant context documents
            
        Returns:
            Dictionary containing answer and citations
        """
        try:
            logger.info("Generating policy answer...")
            # Prepare context
            context_text = "\n".join([doc['text'] for doc, _ in context])
            prompt = f"""Based on the following editorial guidelines, answer this question: {question}

Guidelines:
{context_text}

Answer:"""
            
            answer = self._generate(prompt)
            
            # Extract citations
            citations = []
            for doc, score in context:
                if score > 0.7:  # Only include high-confidence citations
                    citations.append({
                        'source': doc.get('source', 'Unknown'),
                        'text': doc['text'][:100] + '...'  # Truncate long citations
                    })
            
            logger.info("Policy answer generated successfully")
            return {
                'answer': answer,
                'citations': citations
            }
        except Exception as e:
            logger.error(f"Error generating policy answer: {str(e)}")
            raise

    def generate_seo_headline(self, article: Dict[str, Any]) -> str:
        """
        Generate an SEO-optimized headline for an article.
        
        Args:
            article: Article dictionary containing content
            
        Returns:
            SEO-optimized headline (max 100 characters)
        """
        try:
            logger.info("Generating SEO headline...")
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

Article Title: {article.get('content_headline', '')}
Article Content: {article.get('body', '')}

SEO Headline:"""
            
            fallback_headline = article.get('content_headline', '').strip()
            body = article.get('body', '').strip()
            # Try to extract a key phrase from the body for fallback
            key_phrase = ''
            if body:
                # Use the first sentence from the body as a key phrase
                match = re.match(r'([^.!?]*[.!?])', body)
                if match:
                    key_phrase = match.group(1).strip()
            
            for attempt in range(3):
                headline = self._generate(prompt, max_length=100, min_length=50)
                headline = headline.strip().strip('"\'')
                # Truncate at 100 chars, but only at a sentence boundary
                if len(headline) > 100:
                    # Try to find a sentence-ending punctuation within 100 chars
                    match = re.match(r'(.{20,100}[.!?])', headline)
                    if match:
                        headline = match.group(1)
                    else:
                        # Otherwise, cut at last space
                        last_space = headline[:100].rfind(' ')
                        if last_space > 0:
                            headline = headline[:last_space]
                        else:
                            headline = headline[:100]
                # Check if headline is a complete sentence and long enough
                is_sentence = bool(re.match(r'^[A-Z].*[.!?]$', headline))
                if len(headline) >= 20 and is_sentence:
                    logger.info(f"SEO headline generated successfully on attempt {attempt+1}")
                    return headline
                logger.warning(f"Generated headline not valid (attempt {attempt+1}): '{headline}'")
            # Fallback: use article title and key phrase
            if fallback_headline and key_phrase and fallback_headline not in key_phrase:
                fallback = f"{fallback_headline}: {key_phrase}"
                if len(fallback) > 100:
                    # Try to cut at a sentence-ending punctuation within 100 chars
                    match = re.match(r'(.{20,100}[.!?])', fallback)
                    if match:
                        fallback = match.group(1)
                    else:
                        last_space = fallback[:100].rfind(' ')
                        if last_space > 0:
                            fallback = fallback[:last_space]
                        else:
                            fallback = fallback[:100]
                fallback = fallback.rstrip('.!?') + '.'
                logger.info("Using fallback headline (title + key phrase)")
                return fallback
            elif fallback_headline:
                logger.info("Using fallback headline (title only)")
                return fallback_headline[:100]
            else:
                logger.error("No valid headline could be generated.")
                return "News Update."
        except Exception as e:
            logger.error(f"Error generating SEO headline: {str(e)}")
            raise

    def generate_twitter_summary(self, article: Dict[str, Any]) -> str:
        """
        Generate a Twitter-friendly summary of an article.
        
        Args:
            article: Article dictionary containing content
            
        Returns:
            Twitter-friendly summary (max 280 characters)
        """
        try:
            logger.info("Generating Twitter summary...")
            prompt = f"""Write a concise, engaging Twitter summary for this article. The summary must:
1. Be clear and informative
2. Include key facts and context
3. Be engaging and shareable
4. Use active voice
5. Include relevant numbers or statistics
6. Be between 200-280 characters
7. End with a period
8. Be a complete sentence
9. Avoid abbreviations unless necessary
10. Maintain journalistic tone

Article Title: {article.get('content_headline', '')}
Article Content: {article.get('body', '')}

Twitter Summary:"""
            
            summary = self._generate(prompt, max_length=280)
            summary = summary.strip().strip('"\'')
            
            # Ensure the summary ends with a period
            if not summary.endswith('.'):
                summary = summary.rstrip('.!?') + '.'
            
            # Truncate if necessary, but only at a sentence boundary
            if len(summary) > 280:
                match = re.match(r'(.{100,280}[.!?])', summary)
                if match:
                    summary = match.group(1)
                else:
                    last_space = summary[:280].rfind(' ')
                    if last_space > 0:
                        summary = summary[:last_space] + '.'
                    else:
                        summary = summary[:280].rstrip('.!?') + '.'
            
            logger.info("Twitter summary generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating Twitter summary: {str(e)}")
            raise

    def generate_hashtags(self, text: str, max_hashtags: int = 5) -> List[str]:
        """
        Generate relevant hashtags from the given text.
        
        Args:
            text: Input text to generate hashtags from
            max_hashtags: Maximum number of hashtags to generate
            
        Returns:
            List of generated hashtags
        """
        try:
            logger.info("Generating hashtags...")
            prompt = f"""Generate {max_hashtags} relevant hashtags for this text. The hashtags must:
1. Be relevant to the content
2. Be commonly used on social media
3. Be concise and clear
4. Not include spaces or special characters
5. Start with # symbol
6. Be in title case
7. Be specific but not too niche
8. Include location if relevant
9. Include topic keywords
10. Be between 2-4 words each

Text: {text}

Hashtags:"""
            
            hashtags_text = self._generate(prompt, max_length=100)
            hashtags = []
            
            # Extract hashtags from the generated text
            for line in hashtags_text.split('\n'):
                line = line.strip()
                if line.startswith('#'):
                    # Clean up the hashtag
                    hashtag = line.strip('#').strip()
                    # Convert to title case and remove special characters
                    hashtag = ''.join(c for c in hashtag if c.isalnum() or c.isspace())
                    hashtag = hashtag.title().replace(' ', '')
                    if hashtag:
                        hashtags.append(f"#{hashtag}")
            
            # Ensure we don't exceed max_hashtags
            hashtags = hashtags[:max_hashtags]
            
            # If we don't have enough hashtags, add some generic ones
            if len(hashtags) < max_hashtags:
                generic_hashtags = ['#News', '#Breaking', '#Update', '#Latest', '#Report']
                for hashtag in generic_hashtags:
                    if len(hashtags) < max_hashtags and hashtag not in hashtags:
                        hashtags.append(hashtag)
            
            logger.info(f"Generated {len(hashtags)} hashtags successfully")
            return hashtags
        except Exception as e:
            logger.error(f"Error generating hashtags: {str(e)}")
            raise 