import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import jaccard_score

# Sample data
data = {'Item': ['Idly', 'Chicken Vindaloo', 'Ragi Dosa'],
        'Categories': [['South Indian', 'Protein Rich', 'Breakfast', 'Baked Items'],
                       ['North Indian', 'Punjabi', 'Non-Veg', 'Chicken', 'Protein Rich'],
                       ['South Indian', 'Diabetic Friendly', 'Millet Based', 'Pregnancy Friendly']]}

df = pd.DataFrame(data)

# Split data into features and target
X = df['Item']
y = pd.get_dummies(df['Categories'].apply(pd.Series).stack()).sum(level=0)



# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Convert target to NumPy array
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Build Multi-Output Classifier with RandomForest
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model
model.fit(X_train_tfidf, y_train_np)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test_np, y_pred, zero_division=1))

jaccard = jaccard_score(y_test_np, y_pred, average='samples')

print(f'Jaccard Score: {jaccard}')
