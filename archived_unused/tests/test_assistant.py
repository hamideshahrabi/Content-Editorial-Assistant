import sys
from pathlib import Path
import json

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.preprocessing.text_processor import TextProcessor
from src.retrieval.vector_store import VectorStore
from src.generation.text_generator import TextGenerator

def test_policy_qa():
    """Test policy question answering."""
    print("\n=== Testing Policy Q&A ===")
    
    # Initialize components
    text_processor = TextProcessor()
    vector_store = VectorStore()
    text_generator = TextGenerator()
    
    # Load and process policy document
    policy_text = text_processor.load_text_file("data/policies.txt")
    policy_chunks = text_processor.process_policy_document(policy_text, "CBC Editorial Guidelines")
    vector_store.add_documents(policy_chunks)
    
    # Test question
    question = "What's CBC's guideline on citing anonymous sources?"
    print(f"\nQuestion: {question}")
    
    # Get answer
    context = vector_store.search(question, k=3)
    response = text_generator.answer_policy_question(question, context)
    
    print("\nAnswer:", response['answer'])
    print("\nCitations:")
    for citation in response['citations']:
        print(f"- Source: {citation['source']}")
        print(f"  Text: {citation['text']}")

def test_headline_generation():
    """Test SEO headline generation."""
    print("\n=== Testing Headline Generation ===")
    
    # Initialize components
    text_processor = TextProcessor()
    vector_store = VectorStore()
    text_generator = TextGenerator()
    
    # Load and process sample article
    articles = text_processor.load_json_articles("data/articles.json")
    article = articles[0]  # Use first article for testing
    chunks = text_processor.process_article(article)
    vector_store.add_documents(chunks)
    
    print("\nOriginal Title:", article['content_headline'])
    
    # Generate headline
    headline = text_generator.generate_seo_headline(article)
    print("\nGenerated SEO Headline:", headline)

def test_twitter_summary():
    """Test Twitter summary generation."""
    print("\n=== Testing Twitter Summary ===")
    
    # Initialize components
    text_processor = TextProcessor()
    vector_store = VectorStore()
    text_generator = TextGenerator()
    
    # Load and process sample article
    articles = text_processor.load_json_articles("data/articles.json")
    article = articles[0]  # Use first article for testing
    chunks = text_processor.process_article(article)
    vector_store.add_documents(chunks)
    
    print("\nOriginal Title:", article['content_headline'])
    
    # Generate summary
    summary = text_generator.generate_twitter_summary(article)
    print("\nGenerated Twitter Summary:", summary)
    print("\nCharacter count:", len(summary))

if __name__ == "__main__":
    # Run policy QA test
    test_policy_qa() 