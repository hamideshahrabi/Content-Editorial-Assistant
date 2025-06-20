from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import logging
import re
from typing import List, Dict, Optional
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from src.generation.text_generator_new import TextGenerator
from collections import deque
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("Logging initialized - DEBUG VERSION")

# Force CPU usage and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_mem_eager_sdp = False
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Initialize components
model = None
tokenizer = None
flan_model = None
vector_store = None
articles = []
policies = ""
policy_sections = []
text_generator = None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize conversation history
MAX_HISTORY = 10
conversation_history = {}

def clean_text(text: str) -> str:
    """Clean text by removing HTML entities and extra whitespace."""
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs and clean them."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return [clean_text(p) for p in paragraphs]

def split_policy_sections(policies: str):
    """Split the policies text into sections, each starting with a heading."""
    sections = []
    current_section = []
    current_title = None
    
    for line in policies.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Editorial Guidelines:'):
            if current_title and current_section:
                sections.append({
                    'title': current_title,
                    'text': '\n'.join(current_section)
                })
            current_title = line
            current_section = [line]
        elif current_title:
            current_section.append(line)
    
    if current_title and current_section:
        sections.append({
            'title': current_title,
            'text': '\n'.join(current_section)
        })
    
    return sections

def initialize_components():
    """Initialize all required components for the API."""
    global model, tokenizer, flan_model, vector_store, articles, policies, policy_sections, text_generator
    
    try:
        # Load SentenceTransformer model
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Using CPU")
        
        # Load Flan-T5 model and tokenizer
        logger.info("Loading Flan-T5 model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", local_files_only=True)
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
        articles_file = data_dir / "articles.json"
        if articles_file.exists():
            with open(articles_file, 'r') as f:
                articles = json.load(f)
            logger.info(f"Loaded {len(articles)} articles")
        
        # Load policies
        policies_file = data_dir / "policies.txt"
        if policies_file.exists():
            with open(policies_file, 'r') as f:
                policies = f.read()
            policy_sections = split_policy_sections(policies)
            logger.info(f"Loaded {len(policy_sections)} policy sections")
        
        # Initialize text generator
        text_generator = TextGenerator(flan_model, tokenizer)
        
        # Initialize vector store
        if articles:
            texts = [article['content'] for article in articles]
            embeddings = model.encode(texts)
            vector_store = faiss.IndexFlatL2(embeddings.shape[1])
            vector_store.add(embeddings.astype('float32'))
            logger.info("Vector store initialized with article embeddings")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

class Question(BaseModel):
    question: str
    article_id: Optional[str] = None
    session_id: Optional[str] = None
    format: Optional[str] = None

def get_conversation_history(session_id: str) -> List[Dict]:
    """Get conversation history for a session."""
    return conversation_history.get(session_id, [])

def add_to_history(session_id: str, question: str, response: Dict):
    """Add a question and response to the conversation history."""
    if session_id not in conversation_history:
        conversation_history[session_id] = deque(maxlen=MAX_HISTORY)
    conversation_history[session_id].append({
        'question': question,
        'response': response,
        'timestamp': datetime.now().isoformat()
    })

def get_query_type(question: str, history: List[Dict] = None) -> Dict[str, float]:
    """Determine the type of query based on keywords and context."""
    question = question.lower()
    
    # Check for policy-related keywords
    policy_keywords = ['policy', 'guideline', 'rule', 'standard', 'editorial']
    policy_score = sum(1 for word in policy_keywords if word in question)
    
    # Check for headline-related keywords
    headline_keywords = ['headline', 'title', 'seo', 'optimize']
    headline_score = sum(1 for word in headline_keywords if word in question)
    
    # Check for summary-related keywords
    summary_keywords = ['summary', 'summarize', 'brief', 'twitter', 'tweet']
    summary_score = sum(1 for word in summary_keywords if word in question)
    
    # Normalize scores
    total = policy_score + headline_score + summary_score
    if total == 0:
        return {'policy': 0.33, 'headline': 0.33, 'summary': 0.33}
    
    return {
        'policy': policy_score / total,
        'headline': headline_score / total,
        'summary': summary_score / total
    }

def get_article_specific_response(article_id: str, question: str) -> Dict:
    """Generate a response for an article-specific query."""
    article = next((a for a in articles if a['id'] == article_id), None)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    query_type = get_query_type(question)
    
    if query_type['headline'] > 0.5:
        return {
            'headline': {
                'seo_headline': text_generator.generate_headline(article['content']),
                'social_headline': text_generator.generate_social_headline(article['content'])
            }
        }
    elif query_type['summary'] > 0.5:
        return {
            'summary': text_generator.generate_summary(article['content'])
        }
    else:
        return {
            'answer': text_generator.generate_answer(article['content'], question)
        }

def get_relevant_text(question: str) -> Optional[str]:
    """Get relevant text from policies based on the question."""
    if not policy_sections:
        return None
    
    # Encode question
    question_embedding = model.encode([question])[0]
    
    # Encode policy sections
    section_texts = [section['text'] for section in policy_sections]
    section_embeddings = model.encode(section_texts)
    
    # Find most similar section
    scores = np.dot(section_embeddings, question_embedding)
    best_idx = np.argmax(scores)
    
    return policy_sections[best_idx]['text']

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Editorial Assistant API"}

@app.post("/api/article")
async def article_endpoint(request: Question):
    """Handle article-specific queries."""
    if not request.article_id:
        raise HTTPException(status_code=400, detail="Article ID is required")
    
    response = get_article_specific_response(request.article_id, request.question)
    
    if request.session_id:
        add_to_history(request.session_id, request.question, response)
    
    return response

@app.get("/api/articles")
async def get_articles():
    """Get list of available articles."""
    return [{'id': article['id'], 'title': article['title']} for article in articles]

@app.get("/api/article/{article_id}")
async def get_article(article_id: str):
    """Get a specific article by ID."""
    article = next((a for a in articles if a['id'] == article_id), None)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@app.post("/api/qa")
async def qa_endpoint(request: Question):
    """Handle policy Q&A queries."""
    if request.article_id:
        response = get_article_specific_response(request.article_id, request.question)
    else:
        relevant_text = get_relevant_text(request.question)
        if not relevant_text:
            raise HTTPException(status_code=404, detail="No relevant policy found")
        
        response = {
            'answer': text_generator.generate_answer(relevant_text, request.question),
            'citations': [{'source': 'Editorial Guidelines', 'text': relevant_text}]
        }
    
    if request.session_id:
        add_to_history(request.session_id, request.question, response)
    
    return response

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_components() 