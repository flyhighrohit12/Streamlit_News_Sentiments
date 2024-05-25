import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure that the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch news
def fetch_news(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevance',
        'apiKey': 'YOUR_API_KEY'  # Replace with your actual News API key
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to analyze sentiments of news titles
def analyze_sentiments(articles):
    titles = [article['title'] for article in articles]
    sentiments = [sia.polarity_scores(title)['compound'] for title in titles]
    return titles, sentiments

# Streamlit user interface
st.title('Indian News Sentiment Tracker')
query = st.text_input("Enter topic of interest (e.g., 'Indian elections', 'COVID-19 in India', 'Indian economy')", 'Indian elections')

# Fetch and process news articles
if st.button('Analyze Sentiments'):
    news = fetch_news(query)
    if 'articles' in news:
        articles = news['articles']
        titles, sentiments = analyze_sentiments(articles)
        df = pd.DataFrame({'Title': titles, 'Sentiment': sentiments})
        st.write("Sentiment scores range from -1 (most negative) to 1 (most positive).")
        st.dataframe(df)
        
        # Plotting sentiment trends
        fig = px.bar(df, x='Title', y='Sentiment', color='Sentiment',
                     labels={'Title': 'News Article', 'Sentiment': 'Sentiment Score'},
                     color_continuous_scale=px.colors.diverging.Temps, title="News Sentiment Analysis")
        fig.update_layout(xaxis_title="",
                          yaxis_title="Sentiment Score",
                          coloraxis_showscale=True)
        st.plotly_chart(fig)
    else:
        st.error("Failed to fetch news or no articles found.")
