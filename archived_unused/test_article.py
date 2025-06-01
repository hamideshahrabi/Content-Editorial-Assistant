import requests
import json

def test_article():
    article_id = "1.6636959"
    base_url = "http://localhost:8003"
    
    # Test headline generation
    print("\n=== Testing Headline Generation ===")
    headline_response = requests.post(
        f"{base_url}/api/article",
        json={
            "question": "Suggest an SEO-optimized headline",
            "article_id": article_id
        }
    )
    if headline_response.status_code == 200:
        print("\nGenerated Headline:")
        print(json.dumps(headline_response.json(), indent=2))
    else:
        print(f"Error generating headline: {headline_response.text}")
    
    # Test summary generation
    print("\n=== Testing Summary Generation ===")
    summary_response = requests.post(
        f"{base_url}/api/article",
        json={
            "question": "Summarize this article for a Twitter post",
            "article_id": article_id
        }
    )
    if summary_response.status_code == 200:
        print("\nGenerated Summary:")
        print(json.dumps(summary_response.json(), indent=2))
    else:
        print(f"Error generating summary: {summary_response.text}")

if __name__ == "__main__":
    test_article() 