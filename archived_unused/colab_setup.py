# CBC Editorial Assistant - Colab Setup
# Copy and paste these cells into a new Colab notebook

# Cell 1: Mount Google Drive and set up project
from google.colab import drive
import os
from pathlib import Path
import json

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Create project directory in Google Drive
project_dir = '/content/drive/MyDrive/editorial_assistant'
os.makedirs(f'{project_dir}/data', exist_ok=True)
os.makedirs(f'{project_dir}/src/api', exist_ok=True)

# Create requirements.txt
requirements = """fastapi>=0.104.1
uvicorn>=0.24.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
transformers>=4.35.2
torch>=2.1.1
numpy>=1.26.0
pydantic>=2.7.4
requests>=2.31.0
python-multipart>=0.0.6"""

with open(f'{project_dir}/requirements.txt', 'w') as f:
    f.write(requirements)

# Create articles.json
articles = [
    {
        "content_id": "article1",
        "content_headline": "CBC N.L. launches campaign to support food banks",
        "content_type": "news",
        "content_publish_time": "2024-01-15T10:00:00Z",
        "content_last_update": "2024-01-15T10:00:00Z",
        "content_categories": ["Community", "Food Security"],
        "content_tags": ["food bank", "community support", "CBC N.L."],
        "content_word_count": 450,
        "content_department_path": "/news/local",
        "body": "CBC N.L. has launched a campaign to support food banks across the province. The initiative, in partnership with the Community Food Sharing Association, aims to raise awareness and collect donations for food banks that are seeing increased demand. With rising food costs and economic challenges, many families are turning to food banks for support. The campaign will run for the next month, with special programming and community events planned throughout the province."
    },
    {
        "content_id": "article2",
        "content_headline": "New guidelines for social media reporting",
        "content_type": "policy",
        "content_publish_time": "2024-01-10T14:30:00Z",
        "content_last_update": "2024-01-10T14:30:00Z",
        "content_categories": ["Policy", "Social Media"],
        "content_tags": ["social media", "reporting guidelines", "journalism"],
        "content_word_count": 600,
        "content_department_path": "/news/policy",
        "body": "CBC has released updated guidelines for social media reporting. The new policy emphasizes accuracy, transparency, and responsible use of social media platforms. Journalists are required to verify information from social media sources before reporting, maintain professional boundaries, and clearly distinguish between personal and professional accounts. The guidelines also address the use of user-generated content and the importance of protecting sources who share information through social media."
    }
]

articles_path = f'{project_dir}/data/articles.json'
print("\nWriting articles.json...")
with open(articles_path, 'w') as f:
    json.dump(articles, f, indent=2)

# Verify articles.json content
print("\nVerifying articles.json content...")
with open(articles_path, 'r') as f:
    content = f.read()
    print(f"Content length: {len(content)} characters")
    print(f"First 100 characters: {content[:100]}")
    print(f"Last 100 characters: {content[-100:]}")
    # Verify JSON is valid
    try:
        json.loads(content)
        print("JSON is valid")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {str(e)}")

# Create policies.txt
policies = '''CBC Editorial Guidelines: Anonymous Sources

Anonymous sources should only be used when:
1. The information is of significant public interest
2. The source would face serious consequences if identified
3. The information cannot be obtained through on-the-record sources

When using anonymous sources:
- Verify the information through multiple sources
- Clearly explain to readers why anonymity was granted
- Use descriptive terms (e.g., "senior government official" instead of just "source")
- Document the source's credentials and relationship to the story

CBC Editorial Guidelines: Headlines

Headlines should:
1. Be accurate and reflect the content
2. Avoid sensationalism
3. Use clear, concise language
4. Include relevant keywords for SEO
5. Follow CBC's style guide for capitalization and punctuation

CBC Editorial Guidelines: Social Media

When creating content for social media:
1. Keep summaries concise and engaging
2. Use appropriate hashtags
3. Include key information in the first 280 characters
4. Maintain CBC's voice and tone
5. Ensure accuracy and fairness'''

policies_path = f'{project_dir}/data/policies.txt'
print("\nWriting policies.txt...")
with open(policies_path, 'w') as f:
    f.write(policies)

# Create src/api/main.py
main_py_content = '''from fastapi import FastAPI, HTTPException
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
    
    for line in text.split("\\n"):
        if line.startswith("CBC Editorial Guidelines:"):
            if current_section and current_title:
                sections.append({
                    "title": current_title,
                    "content": "\\n".join(current_section).strip()
                })
            current_title = line.strip()
            current_section = []
        else:
            current_section.append(line)
    
    if current_section and current_title:
        sections.append({
            "title": current_title,
            "content": "\\n".join(current_section).strip()
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

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "ok", "message": "CBC Editorial Assistant API is running"}

@app.post("/api/qa")
async def qa_endpoint(request: Question):
    try:
        # Get the question from the request
        question = request.question
        logger.info(f"Received question: {question}")
        
        # Encode question
        logger.info("Encoding question...")
        question_embedding = model.encode([question])
        
        # Search for similar content
        logger.info("Searching for similar content...")
        D, I = vector_store.search(question_embedding.astype("float32"), k=3)
        
        # Get relevant text
        logger.info("Getting relevant text...")
        citations = []
        
        for idx in I[0]:
            source = app.state.sources[idx]
            text = app.state.texts[idx]
            
            # For articles, try to find the most relevant paragraph
            if source["type"] == "article":
                paragraphs = text.split("\n\n")
                most_relevant_para = max(paragraphs, key=lambda p: model.encode([p])[0] @ question_embedding[0])
                citations.append({
                    "source": f"CBC Article: {source['title']}",
                    "text": most_relevant_para
                })
            else:
                # For policies, return the relevant section
                citations.append({
                    "source": source["title"],
                    "text": text
                })
        
        if not citations:
            logger.warning("No relevant information found")
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
'''

# Write the updated main.py
main_py_path = f'{project_dir}/src/api/main.py'
print("\nUpdating main.py...")
with open(main_py_path, 'w') as f:
    f.write(main_py_content)
print("main.py updated successfully")

# Create __init__.py files
with open(f'{project_dir}/src/__init__.py', 'w') as f:
    f.write('')

with open(f'{project_dir}/src/api/__init__.py', 'w') as f:
    f.write('')

# Create a symbolic link to the project directory
os.system(f'ln -s {project_dir} /content/editorial_assistant')

print("\nCreated project structure in Google Drive:")
print(f"Project directory: {project_dir}")
print("\nVerifying directory structure:")
print(f"Directory contents: {os.listdir(project_dir)}")
print(f"src directory contents: {os.listdir(f'{project_dir}/src')}")
print(f"src/api directory contents: {os.listdir(f'{project_dir}/src/api')}")
print(f"data directory contents: {os.listdir(f'{project_dir}/data')}")

# Cell 1.5: Verify data files (run this after creating the files)
print("\nChecking directory contents:")
os.system(f'ls -l {project_dir}/data/')
os.system(f'ls -l {project_dir}/src/api/')

# Cell 2: Install dependencies
print("Installing dependencies...")
os.system('pip install -r requirements.txt')

# Cell 3: Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Set environment variable for Colab
import os
os.environ["COLAB"] = "1"

# Cell 5: Start FastAPI server
print("Starting FastAPI server...")
print("Project directory:", project_dir)

# Print debug information
print("\nDebug Information:")
print(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"Current directory: {os.getcwd()}")
print(f"Project directory contents: {os.listdir(project_dir)}")
print(f"Data directory contents: {os.listdir(f'{project_dir}/data')}")

# Verify articles.json exists and is valid
articles_path = f'{project_dir}/data/articles.json'
print("\nVerifying articles.json...")
if os.path.exists(articles_path):
    try:
        with open(articles_path, 'r') as f:
            content = f.read()
            print(f"Articles file content length: {len(content)}")
            print(f"First 100 characters: {content[:100]}")
            print(f"Last 100 characters: {content[-100:]}")
            json.loads(content)  # Verify JSON is valid
            print("Articles.json is valid")
    except Exception as e:
        print(f"Error reading articles.json: {str(e)}")
else:
    print("Error: articles.json not found!")

# Kill any existing process on port 8000
print("\nKilling any existing process on port 8000...")
os.system('fuser -k 8000/tcp')
time.sleep(2)  # Wait for port to be freed

# Start the server with logging
print("\nStarting server...")
log_file = f'{project_dir}/server.log'

# Start the main server
print("\nStarting main server...")
server_cmd = f'cd {project_dir} && PYTHONPATH={project_dir} python src/api/main.py > {log_file} 2>&1'
os.system(f'{server_cmd} &')

# Wait for server to initialize
print("\nWaiting for server to initialize (30 seconds)...")
time.sleep(30)

# Check if server is running and show logs
try:
    import requests
    response = requests.get("http://localhost:8000/", timeout=5)
    if response.status_code == 200:
        print("Server started successfully!")
        print("You can now run the test cell.")
    else:
        print(f"Warning: Server returned status code {response.status_code}")
        print("\nServer logs:")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                print(f.read())
        else:
            print("No log file found")
except Exception as e:
    print(f"Warning: Could not verify server status - {str(e)}")
    print("\nServer logs:")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            print(f.read())
    else:
        print("No log file found")

# Cell 6: Test the API
print("Starting API test...")

# Test questions
test_questions = [
    # Policy questions
    "What are the guidelines for using anonymous sources in breaking news?",
    "How should journalists handle social media content from anonymous sources?",
    
    # Article-specific questions
    "What is the goal of the Make the Season Kind campaign?",
    "How is the Community Food Sharing Association involved in the campaign?",
    
    # Mixed content questions
    "What are the guidelines for writing headlines about food bank campaigns?",
    "How should social media be used to promote community support initiatives?",
    
    # Specific detail questions
    "What was the amount raised during Feed N.L. Day in 2020?",
    "What percentage increase in need is expected this year for the Salvation Army?"
]

def test_qa_endpoint(question):
    print(f"\n{'='*80}")
    print(f"Testing question: {question}")
    print(f"{'='*80}")
    
    try:
        print("\nSending question to QA endpoint...")
        print(f"Request URL: http://localhost:8000/api/qa")
        
        response = requests.post(
            "http://localhost:8000/api/qa",
            json={"question": question},
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print("\nAnswer:")
            print("-" * 40)
            print(result["answer"])
            print("\nCitations:")
            print("-" * 40)
            for i, citation in enumerate(result["citations"], 1):
                print(f"\nCitation {i}:")
                print(f"Source: {citation['source']}")
                print(f"Text: {citation['text'][:200]}...")
                
            # Evaluate response quality
            print("\nResponse Quality:")
            print("-" * 40)
            if len(result["answer"]) > 500:
                print("⚠️ Warning: Answer is too long")
            if len(result["citations"]) < 1:
                print("⚠️ Warning: No citations provided")
            if len(result["citations"]) > 3:
                print("⚠️ Warning: Too many citations")
        else:
            print(f"\nError: Server returned status code {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"\nError decoding response: {str(e)}")
        print("Response:", response.text)

# Run test questions
print("\nRunning test questions...")
for question in test_questions:
    test_qa_endpoint(question)
    time.sleep(2)  # Small delay between questions

print("\nTest completed.")

# Cell 7: Create Streamlit UI for Colab
print("Creating Streamlit UI for Colab...")

# Create app.py for Streamlit UI
streamlit_app = '''import streamlit as st
import requests
import json
import os

# Set page config
st.set_page_config(
    page_title="CBC Editorial Assistant",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .stButton > button {
        width: 100%;
        font-size: 18px;
        background-color: #FF0000;
        color: white;
    }
    .stButton > button:hover {
        background-color: #CC0000;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📰 CBC Editorial Assistant")
st.markdown("""
    Ask questions about CBC's editorial guidelines, policies, and articles.
    The assistant will provide relevant information with citations.
""")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to get answer from API
def get_answer(question):
    try:
        # Use localhost in Colab
        response = requests.post(
            "http://localhost:8000/api/qa",
            json={"question": question},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "citations" in message:
            with st.expander("View Citations"):
                for i, citation in enumerate(message["citations"], 1):
                    st.markdown(f"**Citation {i}:**")
                    st.markdown(f"**Source:** {citation['source']}")
                    st.markdown(f"**Text:** {citation['text']}")

# Question input
question = st.chat_input("Ask a question about CBC guidelines or articles...")

if question:
    # Add user question to chat
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Get and display answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer(question)
            
            if "error" in response:
                st.error(response["error"])
            else:
                st.write(response["answer"])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response["citations"]
                })
                
                # Display citations in an expander
                with st.expander("View Citations"):
                    for i, citation in enumerate(response["citations"], 1):
                        st.markdown(f"**Citation {i}:**")
                        st.markdown(f"Source: {citation['source']}")
                        st.markdown(f"Text: {citation['text'][:200]}...")
'''

# Create app.py
with open(f'{project_dir}/app.py', 'w') as f:
    f.write(streamlit_app)

# Add requirements for Streamlit
requirements = """streamlit>=1.31.0
requests>=2.31.0"""

with open(f'{project_dir}/requirements.txt', 'a') as f:
    f.write("\n" + requirements)

print("""
To run the Streamlit UI in Colab:

1. First, make sure the FastAPI server is running (Cell 5)

2. Then run these commands in a new cell:
   !pip install streamlit
   !streamlit run app.py

3. Click on the URL that appears in the output (usually http://localhost:8501)

Note: You'll need to run this in a new Colab notebook since we can't run Streamlit in the same notebook as the FastAPI server.
""")

# Add this cell to check the file content
print("Checking current content of main.py in Google Drive...")
main_py_path = '/content/drive/MyDrive/editorial_assistant/src/api/main.py'
try:
    with open(main_py_path, 'r') as f:
        content = f.read()
        print(f"File exists and contains {len(content)} characters")
        print("\nFirst 100 characters:")
        print(content[:100])
except FileNotFoundError:
    print("File not found at:", main_py_path)

# Add this cell to test file updates
print("Testing file update...")
test_content = "# Test update\n" + content
try:
    with open(main_py_path, 'w') as f:
        f.write(test_content)
    print("File updated successfully")
    
    # Verify the update
    with open(main_py_path, 'r') as f:
        new_content = f.read()
        print("\nFirst 100 characters after update:")
        print(new_content[:100])
except Exception as e:
    print("Error updating file:", str(e)) 