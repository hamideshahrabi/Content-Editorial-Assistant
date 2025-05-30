from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import logging
import re
from typing import List, Dict
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import itertools
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', mode='w'),  # Use 'w' mode to overwrite previous logs
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to DEBUG level
logger.debug("Logging initialized - DEBUG VERSION")

# Force CPU usage and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely disable CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_mem_eager_sdp = False
torch.cuda.is_available = lambda: False  # Force CPU-only mode
torch.cuda.device_count = lambda: 0  # Force CPU-only mode

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model = None
tokenizer = None
flan_model = None
vector_store = None
articles = []
policies = ""

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Clean text by removing HTML entities and extra whitespace."""
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs and clean them."""
    # Split on newlines and filter out empty paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    # Clean each paragraph
    return [clean_text(p) for p in paragraphs]

def extract_specific_details(text: str, question: str) -> str:
    """Extract specific details like numbers and percentages from text based on the question."""
    # Look for dollar amounts
    if "amount" in question.lower() or "raised" in question.lower():
        dollar_matches = re.findall(r'\$[\d,]+', text)
        if dollar_matches:
            return f"The amount raised was {dollar_matches[0]}"
    
    # Look for percentages
    if "percentage" in question.lower() or "increase" in question.lower():
        percent_matches = re.findall(r'(\d+)\s*per\s*cent', text, re.IGNORECASE)
        if percent_matches:
            return f"The expected increase is {percent_matches[0]}%"
    
    return text

def extract_full_policy_section(policies: str, heading: str) -> str:
    """Extract the full policy section given a heading from the policies text."""
    lines = policies.split('\n')
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == heading.strip().lower():
            start_idx = i
            break
    if start_idx is None:
        return heading  # fallback: just return the heading
    # Find the next heading or end of file
    section_lines = [lines[start_idx]]
    for line in itertools.islice(lines, start_idx + 1, None):
        if line.strip().startswith('CBC Editorial Guidelines:') and line.strip() != heading.strip():
            break
        section_lines.append(line)
    return '\n'.join(section_lines).strip()

def split_policy_sections(policies: str):
    """Split the policies text into sections, each starting with a heading."""
    sections = []
    current_section = []
    current_title = None
    
    for line in policies.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('CBC Editorial Guidelines:'):
            # Save previous section if exists
            if current_title and current_section:
                sections.append({
                    'title': current_title,
                    'text': '\n'.join(current_section)
                })
            # Start new section
            current_title = line
            current_section = [line]
        elif current_title:
            current_section.append(line)
    
    # Add the last section
    if current_title and current_section:
        sections.append({
            'title': current_title,
            'text': '\n'.join(current_section)
        })
    
    return sections

def initialize_components():
    """Initialize all required components for the API."""
    global model, tokenizer, flan_model, vector_store, articles, policies, policy_sections
    
    try:
        # Load SentenceTransformer model
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("SentenceTransformer model loaded successfully")
        
        # Load Flan-T5 model and tokenizer
        logger.info("Loading Flan-T5 model and tokenizer...")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", local_files_only=True)
        if flan_model is None:
            flan_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-large",
                local_files_only=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
        logger.info("Flan-T5 model and tokenizer loaded successfully")
        
        # Load data
        logger.info("Loading data...")
        data_dir = Path("data")
        logger.info(f"Data directory: {data_dir.absolute()}")
        
        # Load articles
        logger.info("Loading articles...")
        with open(data_dir / "articles.json", "r") as f:
            articles = json.load(f)
        logger.info(f"Successfully loaded {len(articles)} articles")
        
        # Load policies
        logger.info("Loading policies...")
        with open(data_dir / "policies.txt", "r") as f:
            policies = f.read()
        logger.info("Policies loaded")
        
        # Split policies into sections
        policy_sections = split_policy_sections(policies)
        logger.info(f"Split policies into {len(policy_sections)} sections")
        
        # Create vector store
        logger.info("Creating vector store...")
        texts = []
        sources = []
        
        # Add articles (paragraphs)
        for article in articles:
            paragraphs = split_into_paragraphs(article["body"])
            texts.extend(paragraphs)
            sources.extend([{
                "type": "article",
                "title": article["content_headline"]
            } for _ in paragraphs])
        
        # Add policy sections
        for section in policy_sections:
            texts.append(section["text"])
            sources.append({
                "type": "policy",
                "title": section["title"]
            })
        
        embeddings = model.encode(texts)
        logger.info(f"Created embeddings of shape: {embeddings.shape}")
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        vector_store = faiss.IndexFlatL2(dimension)
        vector_store.add(embeddings.astype("float32"))
        logger.info("Vector store created")
        
        # Store sources and texts for later use
        app.state.sources = sources
        app.state.texts = texts
        app.state.policy_sections = policy_sections  # Store policy sections for reference
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "ok", "message": "CBC Editorial Assistant API is running"}

def structure_policy_answer(policy_text: str, question: str) -> str:
    """Fallback function to structure policy answers when model fails."""
    logger.info("Fallback function called with policy text: %s", policy_text)
    # Extract key points from policy text
    lines = policy_text.split('\n')
    summary = ""
    requirements = []
    notes = []
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip the header
        if line.startswith('CBC Editorial Guidelines:'):
            continue
            
        # Extract requirements (numbered or bulleted points)
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '- ', '• ')):
            requirements.append(line)
        # Extract notes (other important information)
        elif ':' in line or 'when' in line.lower():
            notes.append(line)
        # Use first non-empty line as summary
        elif not summary:
            summary = line
    
    # If no summary was found, create one from the first requirement
    if not summary and requirements:
        summary = requirements[0].lstrip('123456789.- ')
    
    # If no requirements were found, use the first note
    if not requirements and notes:
        requirements = [notes[0]]
        notes = notes[1:]
    
    # Format the answer
    formatted_answer = f"""SUMMARY:
{summary}

KEY REQUIREMENTS:
{chr(10).join(f'• {req.lstrip("123456789.- ")}' for req in requirements)}

ADDITIONAL NOTES:
{chr(10).join(f'• {note}' for note in notes)}"""

    logger.info("Formatted answer from fallback: %s", formatted_answer)
    return formatted_answer

@app.post("/api/qa")
async def qa_endpoint(request: Question):
    logger.info("QA endpoint called - DEBUG VERSION")
    try:
        question = request.question
        logger.info(f"Received question: {question}")
        
        # Encode question
        question_embedding = model.encode([question])
        
        # Search for similar content
        logger.info("Searching for similar content...")
        D, I = vector_store.search(question_embedding.astype("float32"), k=5)
        
        # Get relevant text
        logger.info("Getting relevant text...")
        citations = []
        seen_sources = set()
        policy_section = None
        
        # First try to find a relevant policy section
        for idx in I[0]:
            source = app.state.sources[idx]
            if source["type"] == "policy":
                # Find the full policy section
                for section in app.state.policy_sections:
                    if section["title"] == source["title"]:
                        policy_section = {
                            "source": section["title"],
                            "text": section["text"]
                        }
                        break
                if policy_section:
                    break
        
        # If no policy section found, look for relevant articles
        if not policy_section:
            for idx in I[0]:
                source = app.state.sources[idx]
                if source["type"] == "article":
                    source_key = f"{source['type']}:{source['title']}"
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    
                    text = app.state.texts[idx]
                    relevance_score = model.encode([text])[0] @ question_embedding[0]
                    if relevance_score > 0.3:
                        citations.append({
                            "source": f"CBC Article: {source['title']}",
                            "text": text
                        })
                    if len(citations) >= 3:
                        break
        
        if not citations and not policy_section:
            logger.warning("No relevant information found")
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # For policy questions, use the fallback function directly
        if policy_section:
            logger.info("Policy question detected - using fallback formatting directly")
            try:
                answer = structure_policy_answer(policy_section['text'], question)
                logger.info(f"Answer after fallback: {answer}")
            except Exception as e:
                logger.error(f"Error in fallback formatting: {str(e)}")
                answer = "I apologize, but I couldn't generate a properly formatted answer. Please try rephrasing your question."
        else:
            # For non-policy questions, use the model
            # Create prompt for Flan-T5
            few_shot_examples = '''
QUESTION: What are the guidelines for using anonymous sources?
SUMMARY:
Anonymous sources should only be used when the information is of significant public interest, the source would face serious consequences if identified, and the information cannot be obtained through on-the-record sources.
KEY REQUIREMENTS:
• Use anonymous sources only when necessary
• Verify information through multiple sources
• Clearly explain to readers why anonymity was granted
ADDITIONAL NOTES:
• Use descriptive terms for sources
• Document the source's credentials
Policy Guidelines:
CBC Editorial Guidelines: Anonymous Sources\nAnonymous sources should only be used when:\n1. The information is of significant public interest\n2. The source would face serious consequences if identified\n3. The information cannot be obtained through on-the-record sources\nWhen using anonymous sources:\n- Verify the information through multiple sources\n- Clearly explain to readers why anonymity was granted\n- Use descriptive terms (e.g., "senior government official" instead of just "source")\n- Document the source's credentials and relationship to the story

QUESTION: What are the requirements for writing headlines?
SUMMARY:
Headlines must be accurate, fair, and avoid sensationalism. They should reflect the content of the story and not mislead the audience.
KEY REQUIREMENTS:
• Be accurate and fair
• Avoid sensationalism
• Reflect the story content
ADDITIONAL NOTES:
• Headlines should not exaggerate or mislead
Policy Guidelines:
CBC Editorial Guidelines: Headlines\nHeadlines must be accurate, fair and avoid sensationalism. They should reflect the content of the story and not mislead the audience.\nHeadlines should not exaggerate or mislead.'''
            context = "\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in citations])
            prompt = f"""You are a CBC editorial assistant. Answer the following question about CBC's editorial guidelines.
            Your answer MUST follow this exact format:
            
            SUMMARY:
            [Write a 1-2 sentence summary of the key guidelines]
            
            KEY REQUIREMENTS:
            • [First requirement]
            • [Second requirement]
            • [Third requirement]
            
            ADDITIONAL NOTES:
            [Any important exceptions or special cases]
            
            Policy Guidelines:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            # Log the prompt for debugging
            logger.info(f"Prompt sent to model:\n{prompt}")
            
            # Generate answer using Flan-T5
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = flan_model.generate(
                inputs["input_ids"],
                max_length=500,
                num_beams=5,
                temperature=0.5,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                do_sample=True,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.2
            )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Log the raw model output for debugging
            logger.info(f"Raw model output: {answer}")
            
            # Clean up the answer
            answer = answer.strip()
            if not answer or len(answer) < 10:
                answer = "I apologize, but I couldn't generate a proper answer. Please try rephrasing your question."
            
            # Check format for non-policy questions
            logger.info("Checking answer format for non-policy question...")
            required_sections = ["SUMMARY:", "KEY REQUIREMENTS:", "ADDITIONAL NOTES:", "Policy Guidelines:"]
            has_all_sections = all(section in answer for section in required_sections)
            if not has_all_sections:
                logger.info("Answer does not follow required format")
                answer = "I apologize, but I couldn't generate a properly formatted answer. Please try rephrasing your question."
            else:
                logger.info("Answer follows required format")
        
        logger.info(f"Returning answer: {answer}")
        return {
            "answer": answer,
            "citations": [policy_section] if policy_section else citations
        }
    except Exception as e:
        logger.error(f"Error in QA endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_seo_suggestions(article_content: str) -> str:
    """Generate SEO suggestions based on article content using TF-IDF and stop word removal."""
    try:
        # Tokenize and clean the text
        words = word_tokenize(article_content.lower())
        stop_words = set(stopwords.words('english'))
        
        # Remove stop words and non-alphabetic tokens
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        
        # Use TF-IDF to find important keywords
        vectorizer = TfidfVectorizer(
            max_features=10,
            stop_words='english',
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2)  # Include both single words and bigrams
        )
        
        # Create a corpus with the article content
        corpus = [article_content]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the top keywords with their TF-IDF scores
        scores = tfidf_matrix.toarray()[0]
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format the suggestions, focusing on the most relevant terms
        suggestions = []
        for word, score in keyword_scores[:5]:  # Get top 5 keywords
            if score > 0.1:  # Only include terms with significant relevance
                suggestions.append(f"{word} ({score:.2f})")
        
        if suggestions:
            return f"SEO suggestions: {', '.join(suggestions)}"
        else:
            # Fallback to frequency-based approach if no significant terms found
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            suggestions = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return f"SEO suggestions: {', '.join([f'{k} ({v})' for k, v in suggestions])}"
    except Exception as e:
        logger.error(f"Error in SEO suggestions: {str(e)}")
        # Fallback to simple frequency-based approach if TF-IDF fails
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        suggestions = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return f"SEO suggestions: {', '.join([f'{k} ({v})' for k, v in suggestions])}"

def generate_summary(article_content: str) -> str:
    """Generate a concise summary of the article content within 280 characters."""
    try:
        # Use Flan-T5 model for summarization
        prompt = f"""Summarize this article in a concise way (max 280 characters):

{article_content}

Summary:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=280,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure the summary is not too long
        if len(summary) > 280:
            summary = summary[:277] + "..."
        
        return summary
    except Exception as e:
        logger.error(f"Error in summary generation: {str(e)}")
        # Fallback to simple sentence extraction
        sentences = sent_tokenize(article_content)
        summary = ""
        for sentence in sentences[:2]:
            if len(summary + sentence) <= 280:
                summary += sentence + " "
            else:
                break
        return summary.strip()

def generate_headline(article_content: str) -> str:
    """Generate an engaging and SEO-friendly headline based on article content."""
    try:
        # Use Flan-T5 model for headline generation
        prompt = f"""Generate an engaging and SEO-friendly headline for this article (max 100 characters):

{article_content}

Headline:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure the headline is not too long
        if len(headline) > 100:
            headline = headline[:97] + "..."
        
        return headline
    except Exception as e:
        logger.error(f"Error in headline generation: {str(e)}")
        # Fallback to first sentence
        sentences = sent_tokenize(article_content)
        if sentences:
            headline = sentences[0].strip()
            if len(headline) > 100:
                headline = headline[:97] + "..."
            return headline
        return "No headline generated"

# Add new endpoints for SEO suggestions, summaries, and headlines
@app.post("/api/seo")
async def seo_endpoint(request: Question):
    article_content = request.question
    suggestions = generate_seo_suggestions(article_content)
    return {"suggestions": suggestions}

@app.post("/api/summary")
async def summary_endpoint(request: Question):
    article_content = request.question
    summary = generate_summary(article_content)
    return {"summary": summary}

@app.post("/api/headline")
async def headline_endpoint(request: Question):
    article_content = request.question
    headline = generate_headline(article_content)
    return {"headline": headline}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 