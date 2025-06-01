# CBC Editorial Assistant

An AI-powered editorial assistant that helps journalists and editors follow CBC's editorial guidelines and improve their content.

## Features

- **Policy Q&A**: Get answers about CBC's editorial guidelines using RAG (Retrieval-Augmented Generation)
- **SEO Suggestions**: Generate SEO-optimized headlines and content
- **Content Summarization**: Create concise summaries of articles
- **Headline Generation**: Generate engaging and SEO-friendly headlines
- **Twitter Summaries**: Create social media-friendly summaries

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Start the server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8002
```

## Usage

### Interactive Demo
Run the interactive demo:
```bash
python test_demo.py
```

Available commands:
- Type a number (1-100) to view an article
- Type 'generate' followed by the article number to generate content
- Type 'example' to see an example interaction
- Type 'exit' to quit

### API Endpoints

1. Generate Headlines:
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

2. Policy Questions:
```bash
curl -X 'POST' \
  'http://localhost:8002/api/qa' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What are CBCs guidelines on using social media?"
  }'
```

3. Twitter Summaries:
```bash
curl -X 'POST' \
  'http://localhost:8002/api/qa' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Generate a Twitter summary"
  }'
```

## Project Structure

```
editorial_assistant/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── generation/
│   ├── preprocessing/
│   └── retrieval/
├── tests/
│   ├── test_api_qa.py
│   ├── test_article.py
│   └── test_model.py
├── test_demo.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 