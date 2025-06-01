import time
from src.generation.text_generator import TextGenerator
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_articles():
    """Load articles from the data directory"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    articles_file = os.path.join(data_dir, 'articles.json')
    with open(articles_file, 'r') as f:
        return json.load(f)

def format_chat_message(role, content):
    """Format a message in a chat-like style"""
    if role == "user":
        return f"\n👤 User: {content}"
    else:
        return f"\n🤖 Assistant: {content}"

def show_article(article_num, articles):
    """Display an article by its reference number"""
    if 1 <= article_num <= len(articles):
        article = articles[article_num - 1]
        print(format_chat_message("assistant", f"Article #{article_num}:"))
        print(f"\n📰 Headline: {article['content_headline']}")
        print(f"\n📝 Content: {article['body']}\n")
    else:
        print(format_chat_message("assistant", f"Sorry, article #{article_num} not found. Please enter a number between 1 and {len(articles)}"))

def show_example_interaction():
    """Show an example interaction with the chatbot"""
    print("\n" + "="*80)
    print("Example Interaction:")
    print("="*80)
    
    # Example 1: View an article
    print(format_chat_message("user", "1"))
    print(format_chat_message("assistant", "Article #1:"))
    print("\n📰 Headline: Food Bank Sees Record Demand as Winter Approaches")
    print("\n📝 Content: The local food bank is experiencing unprecedented demand as winter approaches, with a 40% increase in visitors compared to last year...\n")
    
    # Example 2: Generate content
    print(format_chat_message("user", "generate 1"))
    print(format_chat_message("assistant", "Generating content for Article #1..."))
    print(format_chat_message("assistant", "SEO Headline: Record-Breaking Food Bank Demand: 40% Surge as Winter Looms"))
    print(format_chat_message("assistant", "Twitter Summary: Local food bank faces 40% surge in demand as winter approaches. Rising costs and economic challenges create 'perfect storm' for families in need."))
    print(format_chat_message("assistant", "Hashtags: #FoodBank #WinterCrisis #CommunitySupport #FoodInsecurity #LocalNews"))
    
    print("\n" + "="*80 + "\n")

def run_demo():
    print("\n" + "="*80)
    print("🤖 EDITORIAL ASSISTANT CHATBOT")
    print("="*80 + "\n")
    print("Available commands:")
    print("- Type a number (1-100) to view an article")
    print("- Type 'generate' followed by the article number to generate content")
    print("- Type 'example' to see an example interaction")
    print("- Type 'exit' to quit")
    print("="*80 + "\n")
    
    # Initialize the text generator
    print(format_chat_message("assistant", "Initializing Text Generator..."))
    start_time = time.time()
    generator = TextGenerator()
    init_time = time.time() - start_time
    print(format_chat_message("assistant", f"Initialization completed in {init_time:.2f} seconds"))
    
    # Load articles
    articles = load_articles()
    
    while True:
        user_input = input("\n👤 User: ").strip()
        
        if user_input.lower() == 'exit':
            print(format_chat_message("assistant", "Goodbye! 👋"))
            break
            
        if user_input.lower() == 'example':
            show_example_interaction()
            continue
            
        # Check if input is a number (article reference)
        if user_input.isdigit():
            show_article(int(user_input), articles)
            continue
            
        # Check if input is a generate command
        if user_input.lower().startswith('generate'):
            try:
                article_num = int(user_input.split()[-1])
                if 1 <= article_num <= len(articles):
                    article = articles[article_num - 1]
                    print(format_chat_message("assistant", f"Generating content for Article #{article_num}..."))
                    
                    # Generate SEO Headline
                    start_time = time.time()
                    headline = generator.generate_seo_headline(article)
                    print(format_chat_message("assistant", f"SEO Headline: {headline}"))
                    
                    # Generate Twitter Summary
                    summary = generator.generate_twitter_summary(article)
                    print(format_chat_message("assistant", f"Twitter Summary: {summary}"))
                    
                    # Generate Hashtags
                    hashtags = generator.generate_hashtags(summary)
                    print(format_chat_message("assistant", f"Hashtags: {' '.join(hashtags)}"))
                else:
                    print(format_chat_message("assistant", f"Invalid article number. Please enter a number between 1 and {len(articles)}"))
            except ValueError:
                print(format_chat_message("assistant", "Invalid command. Please use 'generate' followed by an article number."))
            continue
            
        print(format_chat_message("assistant", "I'm not sure what you mean. Please try one of the available commands."))

if __name__ == "__main__":
    run_demo() 