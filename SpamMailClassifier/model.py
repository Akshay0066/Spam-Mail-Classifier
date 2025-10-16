import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample dataset
data = {
    'text': [
        'Congratulations! You won a free ticket.',
        'Hey, can we meet tomorrow?',
        'Claim your free prize now!',
        'Your assignment is due tomorrow.',
        'Win cash now!!!',
        'Let’s have lunch today.',
        'You have been selected for a free gift.'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# Text preprocessing
X = df['text']
y = df['label']

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
print("Model trained and saved successfully!")
