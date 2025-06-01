import json
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
vector_store = None
articles = []
policies = []
policy_sections = []

def split_policies(text):
    """Split policies into sections based on headers."""
    sections = []
    current_section = []
    current_title = None
    
    for line in text.split("\n"):
        if line.startswith("CBC Editorial Guidelines:"):
            if current_section and current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_section).strip()
                })
            current_title = line.strip()
            current_section = []
        else:
            current_section.append(line)
    
    if current_section and current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_section).strip()
        })
    
    return sections

def initialize_components():
    global model, vector_store, articles, policies, policy_sections
    
    # Load model
    logger.info("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")
    
    # Load data
    logger.info("Loading data...")
    data_dir = Path("data")
    
    # Load articles
    logger.info("Loading articles...")
    with open(data_dir / "articles.json", "r") as f:
        articles = json.load(f)
    logger.info(f"Successfully loaded {len(articles)} articles")
    
    # Load and split policies
    logger.info("Loading policies...")
    with open(data_dir / "policies.txt", "r") as f:
        policies = f.read()
    policy_sections = split_policies(policies)
    logger.info(f"Split policies into {len(policy_sections)} sections")
    
    # Create vector store
    logger.info("Creating vector store...")
    texts = []
    sources = []
    
    # Add articles
    for article in articles:
        texts.append(article["body"])
        sources.append({
            "type": "article",
            "title": article["content_headline"]
        })
    
    # Add policy sections
    for section in policy_sections:
        texts.append(section["content"])
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
    return True

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "ok", "message": "CBC Editorial Assistant API is running"}

@app.post("/api/qa")
async def answer_policy_question(question: Question):
    global model, vector_store
    
    if model is None or vector_store is None:
        success = initialize_components()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize components")

    try:
        # Encode question
        question_embedding = model.encode([question.question])

        # Search for similar content
        D, I = vector_store.search(question_embedding.astype("float32"), k=5)  # Increased from 3 to 5

        # Get relevant text
        citations = []
        seen_texts = set()  # To avoid duplicate citations
        seen_sources = set()  # To avoid duplicate sources
        
        for idx in I[0]:
            source = app.state.sources[idx]
            text = app.state.texts[idx]
            
            # Skip if we've already seen this text or source
            if text in seen_texts or source["title"] in seen_sources:
                continue
            seen_texts.add(text)
            seen_sources.add(source["title"])
            
            # For articles, try to find the most relevant paragraph
            if source["type"] == "article":
                paragraphs = text.split("\\n\\n")
                most_relevant_para = max(paragraphs, key=lambda p: model.encode([p])[0] @ question_embedding[0])
                
                # Extract specific details if relevant
                details = extract_specific_details(most_relevant_para, question.question)
                if details:
                    most_relevant_para = "\\n".join([most_relevant_para] + details)
                
                # Only add if the paragraph is relevant to the question
                if model.encode([most_relevant_para])[0] @ question_embedding[0] > 0.3:  # Relevance threshold
                    citations.append({
                        "source": f"CBC Article: {source['title']}",
                        "text": most_relevant_para
                    })
            else:
                # For policies, only add if the section is relevant to the question
                if model.encode([text])[0] @ question_embedding[0] > 0.3:  # Relevance threshold
                    citations.append({
                        "source": source["title"],
                        "text": text
                    })

        if not citations:
            raise HTTPException(status_code=404, detail="No relevant information found")

        # Return the most relevant text as the answer
        answer = citations[0]["text"]
        
        # Return the response
        return {
            "answer": answer,
            "citations": citations
        }
    except Exception as e:
        logger.error(f"Error in QA endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 