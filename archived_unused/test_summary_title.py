import json
from src.generation.text_generator import TextGenerator

def generate_hashtags(text, max_hashtags=3):
    """Generate relevant hashtags from the text."""
    # Simple hashtag generation by taking key words
    words = text.lower().split()
    # Remove common words and keep only significant ones
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    significant_words = [word for word in words if word not in common_words and len(word) > 3]
    # Take top words and format as hashtags
    hashtags = ['#' + word.capitalize() for word in significant_words[:max_hashtags]]
    return ' '.join(hashtags)

def test_summary_and_title():
    # Initialize the text generator
    generator = TextGenerator()
    
    # Test cases with different types of content
    test_cases = [
        {
            "name": "Food Bank News",
            "content_headline": "Food Bank Sees Record Demand as Winter Approaches",
            "body": """The local food bank is experiencing unprecedented demand as winter approaches, with a 40% increase in visitors compared to last year. The organization, which serves over 5,000 families monthly, is struggling to keep up with the growing need. "We've never seen numbers like this before," says director Sarah Johnson. "The combination of rising food costs and economic challenges has created a perfect storm." Volunteers are working overtime to sort and distribute donations, while the organization is making urgent appeals for more support. The food bank has also expanded its hours and added mobile distribution points to reach more people in need."""
        },
        {
            "name": "Breaking News",
            "content_headline": "Major Infrastructure Project Announced for Downtown",
            "body": """City officials today announced a $500 million infrastructure project that will transform downtown over the next five years. The plan includes new public transit lines, pedestrian-friendly streets, and green spaces. Mayor Jane Smith called it "a once-in-a-generation opportunity to modernize our city." The project is expected to create 2,000 construction jobs and boost local businesses. Public consultations will begin next month."""
        },
        {
            "name": "Feature Story",
            "content_headline": "Local Artist's Mural Project Brings Community Together",
            "body": """When Maria Rodriguez started painting murals in her neighborhood, she never expected it would spark a city-wide movement. Her vibrant designs, inspired by local history and culture, have now spread to 15 different communities. "Art has the power to transform spaces and bring people together," says Rodriguez. The project has received funding from the city's cultural department and will expand to include youth workshops next year."""
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing {test_case['name']}")
        print(f"{'='*50}")
        
        print("\n=== Title Generation ===")
        try:
            headline = generator.generate_seo_headline(test_case)
            print("\nGenerated Headline:")
            print(headline)
            print(f"\nHeadline length: {len(headline)} characters")
        except Exception as e:
            print(f"Error generating headline: {str(e)}")
        
        print("\n=== Summary Generation ===")
        try:
            summary = generator.generate_twitter_summary(test_case)
            print("\nGenerated Summary:")
            print(summary)
            print(f"\nSummary length: {len(summary)} characters")
            
            # Generate hashtags
            hashtags = generate_hashtags(summary)
            print("\nSuggested Hashtags:")
            print(hashtags)
            print(f"Total length with hashtags: {len(summary + ' ' + hashtags)} characters")
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
        
        print(f"\n{'-'*50}")

if __name__ == "__main__":
    test_summary_and_title() 