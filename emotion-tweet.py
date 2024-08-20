import pandas as pd
from textblob import TextBlob
from transformers import pipeline

# Load your dataset
file_path = 'DAX-200824.xlsx'
df = pd.read_excel(file_path)

# Ensure columns are correctly named
df.columns = ['Timestamp', 'Username', 'Tweet', 'Tweet URL','Additional URL']

# Convert the Timestamp to a datetime object and extract the hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%B %d, %Y at %I:%M%p')
df['Hour'] = df['Timestamp'].dt.hour

# Perform sentiment analysis using TextBlob
df['Polarity'] = df['Tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['Subjectivity'] = df['Tweet'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Load the emotion analysis model
emotion_analyzer = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base',top_k=None)

# Function to get the dominant emotion from the Tweet
def get_dominant_emotion(text):
    result = emotion_analyzer(text[:512])
    print(result)  # Print the result to understand its structure
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        return result[0].get('label', 'neutral')
    else:
        return 'neutral'

# Apply the emotion analysis to each tweet
df['Dominant_Emotion'] = df['Tweet'].apply(lambda x: get_dominant_emotion(str(x)))

# Group by hour and calculate the average polarity, subjectivity, count of tweets, and dominant emotion
sentiment_emotion_by_hour = df.groupby('Hour').agg(
    Average_Polarity=('Polarity', 'mean'),
    Average_Subjectivity=('Subjectivity', 'mean'),
    Number_of_Tweets=('Tweet', 'count'),
    Dominant_Emotion=('Dominant_Emotion', lambda x: x.mode()[0] if not x.mode().empty else 'neutral')
).reset_index()

# Save or display the result
print(sentiment_emotion_by_hour)
sentiment_emotion_by_hour.to_excel("Sentiment_and_Emotion_Analysis_by_Hour.xlsx", index=False)
