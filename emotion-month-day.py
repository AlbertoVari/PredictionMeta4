import pandas as pd
from textblob import TextBlob
from transformers import pipeline

# Load your dataset
file_path = 'Versace.xlsx'
df = pd.read_excel(file_path)

# Ensure columns are correctly named
df.columns = ['Timestamp', 'Username', 'Tweet', 'Tweet URL', 'Additional URL']

# Convert the Timestamp to a datetime object and extract the month, day, and hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%B %d, %Y at %I:%M%p')
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour

# Initialize emotion detection pipeline with Jochen Hartmann,
# "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.

emotion_classifier = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    top_k=None
)

# Function to perform sentiment analysis and emotion detection
def analyze_tweet(tweet):
    # Sentiment analysis
    blob = TextBlob(tweet)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Emotion detection
    emotions = emotion_classifier(tweet)
    
    # Handling possible different structures of emotions
    if isinstance(emotions, list) and len(emotions) > 0 and isinstance(emotions[0], list):
        # If emotions is a list of lists, flatten it
        emotions = [item for sublist in emotions for item in sublist]
    
    # Extract the dominant emotion
    if isinstance(emotions, list) and len(emotions) > 0 and isinstance(emotions[0], dict):
        dominant_emotion = max(emotions, key=lambda x: x['score'])['label']
    else:
        dominant_emotion = 'Unknown'
 
    return pd.Series([sentiment, subjectivity, dominant_emotion])

# Apply the analysis to each tweet

df[['Polarity', 'Subjectivity', 'Emotion']] = df['Tweet'].apply(analyze_tweet)

# Crea una lista per raccogliere i tweet con l'emozione "fear" e i relativi punteggi
fear_tweets = []

# Itera attraverso ogni riga del DataFrame e verifica se l'emozione è "fear"
for index, row in df.iterrows():
    if row['Emotion'] == 'fear':
        # Ottieni la lista di emozioni e punteggi per il tweet corrente
        emotions = emotion_classifier(row['Tweet'])

        # Stampa la struttura di `emotions` per esaminarla
        print(f"Tweet: {row['Tweet']}")
        print(f"Emotions structure: {emotions}\n")

        # Trova il punteggio dell'emozione "fear"
        fear_emotion = next((emotion for emotion in emotions if isinstance(emotion, dict) and emotion.get('label') == 'fear'), None)

        if fear_emotion:
            emotion_score = fear_emotion['score']

            # Aggiungi il tweet, l'emozione e il punteggio alla lista
            fear_tweets.append({
                'Tweet': row['Tweet'],
                'Emotion': row['Emotion'],
                'Score': emotion_score
            })

            # Stampa il tweet e il punteggio
            print(f"Tweet: {row['Tweet']}\nEmotion: {row['Emotion']}\nScore: {emotion_score}\n")

# Converti la lista in un DataFrame
fear_df = pd.DataFrame(fear_tweets)

# Salva il DataFrame in un file Excel
output_file_path = 'fear_tweets_with_scores.xlsx'
fear_df.to_excel(output_file_path, index=False)

print(f"Tutti i tweet con l'emozione 'fear' e i relativi punteggi sono stati salvati in: {output_file_path}")


# Add a column for the number of tweets (simply counting each entry as 1 tweet)
df['TweetCount'] = 1

# Group by Month, Day, and Hour for hourly aggregation within the same day
hourly_aggregation = df.groupby(['Month', 'Day', 'Hour']).agg({
    'Polarity': 'mean',
    'Subjectivity': 'mean',
    'Emotion': lambda x: x.mode()[0] if not x.mode().empty else 'None',
    'TweetCount': 'sum'  # Count the number of tweets
}).reset_index()

# Group by Month and Day for daily aggregation within the same month
daily_aggregation = df.groupby(['Month', 'Day']).agg({
    'Polarity': 'mean',
    'Subjectivity': 'mean',
    'Emotion': lambda x: x.mode()[0] if not x.mode().empty else 'None',
    'TweetCount': 'sum'  # Count the number of tweets
}).reset_index()

# Output results
print("Hourly Aggregation (for each day):")
print(hourly_aggregation)
print("\nDaily Aggregation (for each month):")
print(daily_aggregation)

# Optionally, save the results to new Excel sheets or CSV files
hourly_aggregation.to_excel("hourly_aggregation.xlsx", index=False)
daily_aggregation.to_excel("daily_aggregation.xlsx", index=False)

