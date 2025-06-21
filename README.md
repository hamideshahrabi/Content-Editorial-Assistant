# RAG Editorial Assistant

An AI-powered editorial assistant that helps journalists and content creators follow editorial guidelines and improve their content using RAG (Retrieval-Augmented Generation). This tool is designed for news organizations, publishing companies, and media outlets to ensure content quality and adherence to editorial standards.

## Features

- **Policy Q&A**: Get instant answers about editorial guidelines and policies
- **Headline Generation**: Create SEO-optimized and social media-friendly headlines
- **Content Summarization**: Generate concise summaries for social media and briefs
- **Content Analysis**: Analyze articles for guideline compliance
- **Interactive Demo**: Test the system through an easy-to-use interface

## Technical Choices

### Models
1. **Text Generation**: Flan-T5-large
   - Chosen for its strong performance in text generation tasks
   - Better at understanding context and generating coherent responses
   - Alternative choices beyond Hugging Face:
     - GPT-3.5/4: Better performance but higher cost and API dependency
     - BLOOM: Multilingual but larger resource requirements
     - LLaMA: Open source but requires more computational resources

2. **Semantic Search**: SentenceTransformer (all-MiniLM-L6-v2)
   - Efficient for semantic search with good performance
   - Lightweight and fast inference
   - Alternative choices:
     - BERT: Better accuracy but slower
     - USE (Universal Sentence Encoder): Good for multilingual but larger
     - MPNet: Better performance but more resource-intensive

3. **Vector Store**: FAISS
   - Fast and efficient similarity search
   - Good for large-scale vector operations
   - Alternative choices:
     - Pinecone: Cloud-based but requires subscription
     - Weaviate: More features but more complex setup
     - Milvus: Distributed but requires more infrastructure

### Chunking Method
- Semantic chunking based on paragraphs
- Overlap between chunks to maintain context
- Size: 512 tokens with 50 token overlap

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the interactive demo:
```bash
python test_demo.py
```

3. Or start the API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8002
```

## Sample Test Conversations

### 1. Policy Q&A
```bash
curl -X 'POST' \
  'http://localhost:8002/api/qa' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What are the guidelines on using social media for journalists?"
  }'
```

Response:
```json
{
  "answer": "Journalists must maintain professional boundaries on social media, verify information before sharing, and clearly identify themselves as employees. They should avoid sharing personal opinions on controversial topics and ensure their social media presence aligns with journalistic standards.",
  "citations": [
    {
      "source": "Editorial Guidelines",
      "text": "Social media guidelines section..."
    }
  ]
}
```

### 2. Headline Suggestion
```bash
curl -X 'POST' \
  'http://localhost:8002/api/article' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "article_id": "1.6272172",
    "task": "headline",
    "question": "Generate an SEO-optimized headline"
  }'
```

Response:
```json
{
  "headline": {
    "seo_headline": "Food Bank Demand Soars 40% as Winter Approaches",
    "social_headline": "Local food bank sees 40% increase in demand as winter nears"
  }
}
```

### 3. Tweet-style Summary
```bash
curl -X 'POST' \
  'http://localhost:8002/api/qa' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Generate a Twitter summary"
  }'
```

Response:
```json
{
  "summary": "Food bank demand up 40% as winter approaches. Rising costs and economic challenges create perfect storm for families in need. #FoodBank #WinterCrisis"
}
```

## Demo Video
The system is demonstrated in two parts:

1. Interactive Demo Walkthrough: [Watch Demo](https://www.youtube.com/watch?v=eEIpcCS46Jg)
2. API Testing and Examples: [Watch API Demo](https://www.youtube.com/watch?v=sAtIuBn5A1E)

The videos demonstrate:
- Interactive demo usage
- API endpoint testing
- Real-time content generation
- Policy Q&A examples

## Use Cases

This editorial assistant is ideal for:

- **News Organizations**: Ensure journalists follow editorial standards
- **Publishing Companies**: Maintain consistent content quality across publications
- **Media Outlets**: Streamline content review processes
- **Content Teams**: Generate optimized headlines and summaries
- **Editorial Departments**: Provide quick access to style guides and policies

## Project Structure
```
editorial_assistant/
├── src/
│   ├── api/          # FastAPI endpoints
│   ├── generation/   # Text generation with Flan-T5
│   └── retrieval/    # RAG implementation
├── test_demo.py      # Interactive demo
├── requirements.txt  # Dependencies
└── README.md
```

## License
MIT License
