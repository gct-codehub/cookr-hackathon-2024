import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Sample dataset
import pandas as pd

# Load the dataset
df = pd.read_csv('/content/200_rows_dataset.csv')

# Access the reviews and aspects columns
reviews = df['Review'].tolist()
aspects = df['Aspect'].tolist()


# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
sentiments = [sid.polarity_scores(review)["compound"] for review in reviews]

# Combine reviews and sentiments as features
features = [f"{reviews[i]} {sentiments[i]}" for i in range(len(reviews))]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, aspects, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Creating and training a basic classifier (Multinomial Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predicting the aspects for the test set
y_pred = classifier.predict(X_test_vectorized)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

# Displaying results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

