import pytest
from src.generation.text_generator import TextGenerator

@pytest.fixture
def text_generator():
    return TextGenerator()

@pytest.fixture
def sample_article():
    return {
        'content_headline': 'CBC N.L. launches annual "Make the Season Kind" campaign to support food banks',
        'body': 'CBC Newfoundland and Labrador has launched its annual "Make the Season Kind" campaign to help support food banks across the province. The campaign, which runs until Dec. 31, aims to keep food pantries stocked for those in need. This year\'s campaign is being held in partnership with the Community Food Sharing Association, which distributes food to more than 50 food banks across Newfoundland and Labrador. "We know that food insecurity is a growing issue in our province," said CBC N.L. managing editor Stephanie Kinsella. "This campaign is one way we can help make a difference in our communities." The campaign encourages people to donate non-perishable food items or make monetary donations to their local food bank. Last year, the campaign helped raise more than $50,000 and collected thousands of pounds of food for those in need.',
        'content_categories': [{'content_category': 'News'}, {'content_category': 'Community'}],
        'content_tags': [{'name': 'Food Bank'}, {'name': 'Charity'}]
    }

def test_generate_seo_headline(text_generator, sample_article):
    headline = text_generator.generate_seo_headline(sample_article)
    assert headline
    assert len(headline) >= 20
    assert len(headline) <= 100
    assert 'CBC' in headline or 'food bank' in headline.lower()

def test_generate_twitter_summary(text_generator, sample_article):
    summary = text_generator.generate_twitter_summary(sample_article)
    assert summary
    assert len(summary) >= 180
    assert len(summary) <= 280
    assert 'CBC' in summary or 'food bank' in summary.lower()
    assert '#' in summary  # Should include hashtags 