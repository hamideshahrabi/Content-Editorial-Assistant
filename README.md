# CBC Editorial Assistant

An AI-powered editorial assistant that helps journalists and editors follow CBC's editorial guidelines and improve their content.

## Features

- **Policy Q&A**: Get answers about CBC's editorial guidelines
- **SEO Suggestions**: Generate SEO-optimized keywords and phrases
- **Content Summarization**: Create concise summaries of articles
- **Headline Generation**: Generate engaging and SEO-friendly headlines

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
python src/api/main.py
```

## API Endpoints

- `POST /api/qa`: Get answers about editorial guidelines
- `POST /api/seo`: Generate SEO suggestions
- `POST /api/summary`: Generate article summaries
- `POST /api/headline`: Generate headlines

## Project Structure

```
editorial_assistant/
├── data/
│   ├── articles.json
│   └── policies.txt
├── src/
│   ├── api/
│   │   └── main.py
│   ├── generation/
│   ├── preprocessing/
│   └── retrieval/
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