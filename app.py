import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()

def text_transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english"):
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

mnb = pickle.load(open("mnb.pkl","rb"))
tfidf = pickle.load(open("tfidf.pkl","rb"))

st.title("SMS Spam Classfifer")

input_sms = st.text_area("Enter The Message")

if st.button("Predict"):
    
    transformed_text = text_transformer(input_sms)
    tfidf_transform = tfidf.transform([transformed_text])
    mnb_prediction = mnb.predict(tfidf_transform)[0]

    if mnb_prediction == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
