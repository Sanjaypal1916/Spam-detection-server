import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_test(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)


model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

st.title("SMS Spam Detection")
txt = st.text_input("enter the message here")

if st.button("predict"):

    trans_text = transform_test(txt)
    vector_text = tfidf.transform([trans_text])
    prediction = model.predict(vector_text)[0]

    if prediction > 0.5:
        st.header("its a Spam")
    else:
        st.header("Its a Ham")

