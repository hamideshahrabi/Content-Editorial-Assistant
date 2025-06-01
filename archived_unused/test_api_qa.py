import requests
import json

def test_question(question):
    print("\n" + "="*80)
    print(f"Testing question: {question}")
    print("="*80 + "\n")
    
    response = requests.post(
        "http://localhost:8003/api/qa",
        json={"question": question}
    )
    
    print(f"Response status: {response.status_code}\n")
    if response.status_code == 200:
        result = response.json()
        print("Answer:")
        print("-" * 40)
        if "answer" in result:
            print(result["answer"])
        else:
            print("No 'answer' key in response. Full response:")
            print(json.dumps(result, indent=2))
        print("\nCitations:")
        print("-" * 40)
        if "citations" in result:
            for i, citation in enumerate(result["citations"], 1):
                print(f"\nCitation {i}:")
                print(f"Source: {citation['source']}")
                print(f"Text: {citation['text']}")
        else:
            print("No 'citations' key in response.")

# Test questions
questions = [
    # Policy-related questions
    "What are the guidelines for using social media in news reporting?",
    "How should anonymous sources be handled?",
    "What are the requirements for writing headlines?",
    "What are the guidelines for food bank donations?",
    "How should journalists handle breaking news on social media?",
    "What are the rules for using user-generated content?",
    "How should journalists verify information from social media?",
    
    # Article-specific questions
    "Tell me about the Make the Season Kind campaign",
    "What is the goal of the Community Food Sharing Association?",
    "How has the need for food bank services changed recently?",
    "What role do volunteers play in food bank operations?",
    
    # Headline and summary requests
    "Generate a headline for article 123",
    "Create a Twitter summary for article 456",
    "Write a social media summary for the latest food bank article",
    
    # Complex questions requiring multiple sources
    "How do food bank guidelines align with CBC's community support policies?",
    "What are the best practices for reporting on social issues like food insecurity?",
    "How should journalists balance speed and accuracy when reporting on community events?",
    
    # Specific detail questions
    "What was the amount raised during Feed N.L. Day in 2020?",
    "What percentage increase in need is expected this year for the Salvation Army?",
    "How many volunteers typically help during food bank campaigns?"
]

print("Starting API test...")
print("Testing API at: http://localhost:8003/api/qa\n")

# Check if server is running
try:
    response = requests.get("http://localhost:8003/")
    print("Checking if server is running...")
    if response.status_code == 200:
        print("Server is running!\n")
    else:
        print("Warning: Server returned unexpected status code:", response.status_code)
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to server. Make sure it's running on port 8003.")
    exit(1)

print("Running test questions...\n")

# Run tests
for question in questions:
    test_question(question)

print("\nTest completed.") 