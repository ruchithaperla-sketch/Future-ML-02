import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# download stopwords
nltk.download('stopwords')

# title
st.title("Customer Support Ticket Classification")
st.write("Enter a support ticket and get predicted category")

# load dataset
df = pd.read_csv("customer_support_tickets.csv")

# select columns
df['text'] = df['Ticket Description']
df['category'] = df['Ticket Type']

# stopwords
stop_words = set(stopwords.words('english'))

# clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

y = df['category']

# train model
model = MultinomialNB()
model.fit(X, y)

# user input
ticket = st.text_area("Enter Ticket Description")

# predict button
if st.button("Predict Category"):

    if ticket.strip() == "":
        st.warning("Please enter ticket text")
    else:
        cleaned = clean_text(ticket)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        st.success("Predicted Category:")
        st.write(prediction[0])