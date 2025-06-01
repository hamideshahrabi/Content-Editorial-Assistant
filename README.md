# CBC Editorial Assistant

An AI-powered editorial assistant that helps journalists follow CBC's guidelines and improve their content using RAG (Retrieval-Augmented Generation).

## Features

- **Policy Q&A**: Get answers about CBC's editorial guidelines using RAG
- **SEO Suggestions**: Generate SEO-optimized headlines and content
- **Content Summarization**: Create concise summaries of articles
- **Headline Generation**: Generate engaging and SEO-friendly headlines
- **Twitter Summaries**: Create social media-friendly summaries

## Technical Choices

### Models
1. **Text Generation**: Flan-T5-large
   - Strong performance in text generation tasks
   - Good context understanding and coherence
2. **Semantic Search**: SentenceTransformer (all-MiniLM-L6-v2)
   - Efficient, lightweight, and fast inference
3. **Vector Store**: FAISS
   - Fast and efficient similarity search for large-scale vector operations

### Chunking Method
- Semantic chunking based on paragraphs
- Overlap between chunks to maintain context (512 tokens, 50 token overlap)

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
    "question": "What are CBCs guidelines on using social media?"
  }'
```
Response:
```json
{
  "answer": "CBC journalists must maintain professional boundaries on social media, verify information before sharing, and clearly identify themselves as CBC employees. They should avoid sharing personal opinions on controversial topics and ensure their social media presence aligns with CBC's journalistic standards.",
  "citations": [
    {
      "source": "CBC Editorial Guidelines",
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

## Project Structure
```
editorial_assistant/
├── src/
│   ├── api/
│   │   └── main.py           # Core API implementation
│   └── generation/
│       └── text_generator_new.py  # Optimized text generation
├── data/                     # Data files
│   ├── articles.json
│   └── policies.txt
├── test_demo.py              # Interactive demo
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── .gitignore
```

## License
MIT License 