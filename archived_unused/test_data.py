import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    try:
        # Test articles.json
        logger.info("Testing articles.json loading...")
        with open('data/articles.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)
            logger.info(f"Successfully loaded {len(articles)} articles")
            
        # Test policies.txt
        logger.info("Testing policies.txt loading...")
        with open('data/policies.txt', 'r', encoding='utf-8') as f:
            policies = f.read()
            logger.info(f"Successfully loaded policies.txt ({len(policies)} characters)")
            
        return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False

if __name__ == "__main__":
    test_data_loading() 