import os
import pandas as pd
from google.cloud import language_v1

# Set the environment variable for Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ayunouvalina14/nlp/cloud-computing-429816-28812270bc6c.json"

# Initialize the Natural Language API client
client = language_v1.LanguageServiceClient()

def classify_sentiment(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_sentiment(document=document)
    sentiment = response.document_sentiment
    # Classify sentiment score as 1 (positive) if greater than 0, else 0 (negative)
    sentiment_class = 1 if sentiment.score > 0 else 0
    return sentiment_class

# Load data
data = pd.read_csv('/home/ayunouvalina14/nlp/text-nlp.csv') 

# Create a list to store classification results
sentiment_classes = []

# Process Each Text Entry
for text in data['text']:
    sentiment_class = classify_sentiment(text)
    sentiment_classes.append(sentiment_class)

# Add the Results to the DataFrame
data['sentiment_class'] = sentiment_classes

# Save the results to a new CSV file
data.to_csv('/home/ayunouvalina14/nlp/sentiment_classification_results.csv', index=False)

print("Sentiment classification completed. Results saved to 'sentiment_classification_results.csv'.")