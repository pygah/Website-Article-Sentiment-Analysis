import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the VADER sentiment intensity analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def fetch_article_text(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Raise an exception if the request was unsuccessful
    response.raise_for_status()
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract text from the webpage
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.get_text() for para in paragraphs])
    return article_text

def analyze_article_sentiment(article_text):
    # Split the article into smaller chunks if it's too long
    max_length = 512  # Arbitrary chunk size to mimic the transformers model's token limit
    article_chunks = [article_text[i:i+max_length] for i in range(0, len(article_text), max_length)]
    
    # Analyze sentiment for each chunk
    sentiments = []
    for chunk in article_chunks:
        sentiment_score = sentiment_analyzer.polarity_scores(chunk)
        if sentiment_score['compound'] >= 0.05:
            label = 'POSITIVE'
        elif sentiment_score['compound'] <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        sentiments.append({'chunk': chunk, 'label': label, 'score': sentiment_score})
    
    # Aggregate results
    positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
    negative_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'NEGATIVE')
    
    # Determine overall sentiment
    if positive_count > negative_count:
        overall_sentiment = "Positive"
    elif negative_count > positive_count:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    
    return overall_sentiment, sentiments

# Example usage
url = 'https://weather.com/weather/today/l/4e866bb30db56b0762dd8098335cff1c7b7c02f24d6dfdf41d8765b27cd929d5'
article_text = fetch_article_text(url)
overall_sentiment, detailed_sentiments = analyze_article_sentiment(article_text)
print(f"Overall Sentiment: {overall_sentiment}")
print("Detailed Sentiments:", detailed_sentiments)
