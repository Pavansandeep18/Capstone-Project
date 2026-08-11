import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
pip install gensim
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

file_path = "/content/1785943249747-Amazon_Reviews.csv"

df = pd.read_csv(
    file_path,
    engine="python",
    on_bad_lines="skip"
)

print("Dataset Shape:", df.shape)

df.head()

df = pd.read_csv(
    r"/content/1785943249747-Amazon_Reviews.csv",
    engine="python",
    on_bad_lines="skip"
)

print("Columns:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

df['Review Text'] = df['Review Text'].fillna("")
df['Review Title'] = df['Review Title'].fillna("")
df['Rating'] = df['Rating'].fillna("")

df = df[
    (df['Review Text'].str.strip() != "") &
    (df['Rating'].str.strip() != "")
]

print("Shape after handling missing values:", df.shape)

df['RatingNumeric'] = df['Rating'].str.extract(r'(\d+)').astype(float)

df[['Rating', 'RatingNumeric']].head()

print(df['RatingNumeric'].value_counts().sort_index())

def sentiment_label(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"

df['SentimentLabel'] = df['RatingNumeric'].apply(sentiment_label)

df[['RatingNumeric', 'SentimentLabel']].head()

print(df['SentimentLabel'].value_counts())

df['ReviewLength'] = df['Review Text'].apply(
    lambda x: len(str(x).split())
)

df['CharacterLength'] = df['Review Text'].apply(
    lambda x: len(str(x))
)

df[['Review Text', 'ReviewLength', 'CharacterLength']].head()

df['ReviewDate'] = pd.to_datetime(
    df['Review Date'],
    errors='coerce'
)

df['ReviewYear'] = df['ReviewDate'].dt.year
df['ReviewMonth'] = df['ReviewDate'].dt.month
df['ReviewMonthName'] = df['ReviewDate'].dt.month_name()

df[
    ['ReviewDate', 'ReviewYear', 'ReviewMonth', 'ReviewMonthName']
].head()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

df['CleanedReview'] = df['Review Text'].apply(clean_text)

df[['Review Text', 'CleanedReview']].head()

df.head()

print(df.columns.tolist())

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='RatingNumeric'
)

plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='SentimentLabel'
)

plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')

plt.show()

sentiment_percentage = (
    df['SentimentLabel']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(sentiment_percentage)

country_sentiment = pd.crosstab(
    df['Country'],
    df['SentimentLabel']
)

country_sentiment.head(10)

country_sentiment.plot(
    kind='bar',
    figsize=(12,6)
)

plt.title('Sentiment Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Number of Reviews')

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

country_rating = (
    df.groupby('Country')['RatingNumeric']
    .mean()
    .sort_values(ascending=False)
)

print(country_rating.head(10))

plt.figure(figsize=(12,6))

country_rating.head(20).plot(kind='bar')

plt.title('Top 10 Countries by Average Rating')
plt.xlabel('Country')
plt.ylabel('Average Rating')

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

plt.figure(figsize=(10,5))

sns.histplot(
    data=df,
    x='ReviewLength',
    bins=50
)

plt.title('Review Length Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Number of Reviews')

plt.xlim(0, 500)

plt.show()

df.groupby('SentimentLabel')['ReviewLength'].mean()

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x='SentimentLabel',
    y='ReviewLength'
)

plt.title('Average Review Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Review Length')

plt.show()

year_reviews = df['ReviewYear'].value_counts().sort_index()

print(year_reviews)

plt.figure(figsize=(10,5))

year_reviews.plot(kind='bar')

plt.title('Number of Reviews by Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')

plt.show()

model_df = df[
    df['CleanedReview'].str.strip() != ""
].copy()

X = model_df['CleanedReview']
y = model_df['SentimentLabel']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

y_pred = model.predict(X_test_tfidf)

print(y_pred[:20])

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

new_review = [
    "The product was excellent and delivery was very fast"
]

new_review_clean = [
    clean_text(review)
    for review in new_review
]

new_review_tfidf = tfidf.transform(
    new_review_clean
)

prediction = model.predict(
    new_review_tfidf
)

print("Predicted Sentiment:", prediction[0])

new_review = [
    "Very bad experience. The product was damaged and customer service was terrible."
]

new_review_clean = [
    clean_text(review)
    for review in new_review
]

new_review_tfidf = tfidf.transform(
    new_review_clean
)

prediction = model.predict(
    new_review_tfidf
)

print("Predicted Sentiment:", prediction[0])

import joblib

joblib.dump(
    model,
    'sentiment_model.pkl'
)

joblib.dump(
    tfidf,
    'tfidf_vectorizer.pkl'
)

print("Model saved successfully.")

df.to_csv(
    'Amazon_Reviews_Cleaned.csv',
    index=False
)

print("Cleaned dataset saved.")

top_products = (
    df['productTitle']
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title('Top 10 Most Reviewed Products')
plt.xlabel('Number of Reviews')
plt.ylabel('Product')

plt.show()

category_sentiment = pd.crosstab(
    df['category'],
    df['SentimentLabel']
)

category_sentiment.plot(
    kind='bar',
    figsize=(12,6)
)

plt.title('Sentiment Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Number of Reviews')

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

import streamlit as st
import pandas as pd
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Load model and vectorizer
# -----------------------------

model = joblib.load(
    'sentiment_model.pkl'
)

tfidf = joblib.load(
    'tfidf_vectorizer.pkl'
)

# -----------------------------
# NLTK setup
# -----------------------------

stop_words = set(
    stopwords.words('english')
)

lemmatizer = WordNetLemmatizer()


# -----------------------------
# Text preprocessing
# -----------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    text = re.sub(
        r'\d+',
        '',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title(
    "Amazon Product Review Sentiment Analysis"
)

st.write(
    "Enter an Amazon product review "
    "to predict its sentiment."
)

review = st.text_area(
    "Enter your review:"
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        cleaned_review = clean_text(
            review
        )

        vectorized_review = tfidf.transform(
            [cleaned_review]
        )

        prediction = model.predict(
            vectorized_review
        )[0]

        st.subheader(
            "Prediction"
        )

        st.success(
            f"Sentiment: {prediction}"
        )
