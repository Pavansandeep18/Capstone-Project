Amazon Product Reviews Analytics Capstone Project

Step 1: Importing the Libraries

import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
!pip install gensim
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

....................................................................................

Step 2: Load Dataset

df = pd.read_csv(
    "/content/Amazon_Reviews.csv",
    engine="python",
    encoding="latin1"
)

.......................................................................

Step 3: Check Missing Columns

df.isnull().sum()

....................
Handle Missing Values

df.dropna(subset=[
    'Review Text',
    'Rating',
    'Review Title',
    'Country',
    'Review Date',
    'Date of Experience'
], inplace=True)

df.isnull().sum()
....................................
Check Duplicated Records

print(df.duplicated().sum())

.....................................

Drop Unwanted Columns

df.drop(columns=[
    'Profile Link',
    'Review Count',
    'Date of Experience'
], inplace=True)

..................................

Check Data Types
print(df.dtypes)
.....................................
Convert Rating to numeric

df["Rating"] = df["Rating"].str.extract(r'(\d+\.?\d*)')
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
.........................................................................
Convert Review Date to Datetime

# Convert Review Date to datetime
df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce")

# Extract Year
df["Year"] = df["Review Date"].dt.year

# Extract Month Name
df["Month"] = df["Review Date"].dt.month_name()

# Extract Month Number
df["Month_Number"] = df["Review Date"].dt.month


# Display the result
print(df[["Review Date", "Year", "Month"]].head())

.................................................................

df.columns.to_list()
df.isnull().sum()
print(df["Rating"].head(10))

.....................................................

Step 4: Download NLTK data

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

..............................................

Step 5: Feature Engineering

# Review Length
df["ReviewLength"]=df["Review Text"].apply(lambda x:len(str(x).split()))

.......................

# Sentiment Label
def sentiment(r):

    if r>=4:
        return "Positive"

    elif r<=2:
        return "Negative"

    else:
        return "Neutral"

df["SentimentLabel"]=df["Rating"].apply(sentiment)

..........................................................

Step 6: Text Preprocessing

lemmatizer=WordNetLemmatizer()

stop=set(stopwords.words("english"))

def clean(text):

    text=text.lower()

    text=re.sub(r'[^a-zA-Z ]',' ',text)

    words=text.split()

    words=[lemmatizer.lemmatize(word) for word in words if word not in stop]

    return " ".join(words)

df["CleanReview"]=df["Review Text"].apply(clean)

.............................................................................
Step 7 : Encoding

le=LabelEncoder()

df["Country"]=le.fit_transform(df["Country"])

df["Sentiment"]=le.fit_transform(df["SentimentLabel"])

....................................................

Step 8: TF-IDF

tfidf=TfidfVectorizer(max_features=5000)

X=tfidf.fit_transform(df["CleanReview"])

.......................................................

Step 9: Scaling

scaler=StandardScaler(with_mean=False)

X=scaler.fit_transform(X)
.....................................................

Step 10: Training and Testing

y=df["Sentiment"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

..................................................................................

Train the model

model=LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)

..................................
Step 11 : Prediction

pred=model.predict(X_test)

....................................

Step 12 : Evaluation

print("Accuracy")
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

...............................

#Confusion Matrix
cm=confusion_matrix(y_test,pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')

......................................................
Build Visualizations

# Rating Distribution
plt.figure(figsize=(8,5))

sns.countplot(x="Rating",data=df)

plt.title("Rating Distribution")

plt.show()

...............................................

#Sentiment distribution by category.

plt.figure(figsize=(12,6))

sns.countplot(
x="Country",
hue="SentimentLabel",
data=df)

plt.show()

........................................

#Top 10 most reviewed products using Matplotlib, Seaborn.
top=df['Review Title'].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(x=top.values,y=top.index)

plt.title("Top 10 Most Reviewed Products")

................................................

Save The Model

import joblib
joblib.dump(model,"model.pkl")
joblib.dump(tfidf,"tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")

.....................................

print(le.classes_)

..................................

Save the processed dataset

df.to_csv("processed_reviews.csv", index=False)

...................................................
Streamlit App


%%writefile app.py

import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load saved model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load processed dataset
df = pd.read_csv("processed_reviews.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# TF-IDF matrix for recommendations
review_matrix = tfidf.transform(df["CleanReview"])

st.title("Amazon Product Review Analytics")

review = st.text_area("Enter Product Review")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        vector = tfidf.transform([cleaned_review])

        prediction = model.predict(vector)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Sentiment: {sentiment}")

        similarity = cosine_similarity(vector, review_matrix)
        top3 = similarity.argsort()[0][-3:][::-1]

        st.subheader("Top 3 Similar Products")

        for idx in top3:
            st.write("**Review Title**", df.iloc[idx]["Review Title"])
            st.write("**Country:**", df.iloc[idx]["Country"])
            st.write("**Rating:**", df.iloc[idx]["Rating"])
            st.write(df.iloc[idx]["Review Text"])
            st.write("---")

.....................................................................

!ls

..........................................

Download the file app.py

from google.colab import files
files.download("app.py")