import logging
from src.api.main import initialize_components

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    print("\n=== CBC Editorial Assistant Test Script ===\n")
    
    print("Initializing models and loading data...")
    initialize_components()
    
    # Import components after initialization
    from src.api.main import model, vector_store, articles, policies
    
    while True:
        print("\nEnter your question (or 'quit' to exit):")
        question = input("> ").strip()
        
        if question.lower() == 'quit':
            break
            
        if not question:
            continue
            
        try:
            # Encode question
            question_embedding = model.encode([question])
            
            # Search for similar content
            D, I = vector_store.search(question_embedding.astype("float32"), k=3)
            
            # Get relevant text
            citations = []
            for idx in I[0]:
                if idx < len(articles):
                    citations.append({
                        "source": f"CBC Article: {articles[idx]['content_headline']}",
                        "text": articles[idx]["body"]
                    })
                else:
                    citations.append({
                        "source": "CBC Guidelines",
                        "text": policies
                    })
            
            if not citations:
                print("\nNo relevant information found.")
                continue
                
            # Print the answer and citations
            print("\nAnswer:")
            print(citations[0]["text"])
            
            print("\nCitations:")
            for i, citation in enumerate(citations, 1):
                print(f"\n{i}. Source: {citation['source']}")
                print(f"Text: {citation['text'][:200]}...")
                
        except Exception as e:
            print(f"\nError processing question: {str(e)}")

if __name__ == "__main__":
    setup_logging()
    main() 