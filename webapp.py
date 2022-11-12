import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re,nltk,json
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
from keras.models import load_model

model1 = pk.load(open('DT_model.sav', 'rb'))
model2 = pk.load(open('BetterModel.sav', 'rb'))


st.markdown(
    """
    <style>
    .header-style {
        font-size:25px;
        font-family:sans-serif;
        position:absolute;
        text-align: center;
        color: 032131;
        top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .font-style {
        font-size:20px;
        font-family:sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# st.markdown(
#     "<p class="header-style">Student's Anxiety/Depression Detection Webapp</p>",
#     unsafe_allow_html=True
# )
st.header("Student's Anxiety/Depression Detection Webapp")
st.text("")
st.text("")
st.image('image.jpg',width=700)

#dataset loading
df = pd.read_excel('dataset.xlsx')
df = df[:3000]
df = df.sample(frac = 1)
df.dropna()

lm  = WordNetLemmatizer()
nltk.download('wordnet')

def text_transformation(col):
    corpus = []
    for token in col:
        alphabet = re.sub('[^a-zA-Z]',' ',str(token))
        alphabet = alphabet.lower()
        alphabet = alphabet.split()
        a_lemmas = [lm.lemmatize(word) for word in alphabet if word not in stop_words]
        corpus.append(' '.join(str(x) for x in a_lemmas))
    return corpus


df['cleaned'] = text_transformation(df['text'])

# Feature Extraction
X = df.cleaned
y = df.label.astype(int)
vect = TfidfVectorizer(max_features = 20000 , lowercase=False , ngram_range=(1,2),use_idf = True)
X_tfidf =vect.fit_transform(X).toarray()


def text_cl(raw_text):
    alphabet = re.sub('[^a-zA-Z]',' ',str(raw_text))
    alphabet = alphabet.lower()
    alphabet = alphabet.split()
    a_lemmas = [lm.lemmatize(word) for word in alphabet if word not in stop_words]
    cleant = ' '.join(str(x) for x in a_lemmas)
    return cleant


def Anxiety_detection(text):
    # column_1, column_2,column_3= st.beta_columns(3)
    column_1, column_2 = st.beta_columns(2)
    result1 = model1.predict(vect.transform([text]).toarray())[0]
    # result2 = model2.predict(vect.transform([text]).toarray())[0]
    column_1.write("Model Prediction: ")
    if result1 == 1:
        column_2.write("Anxiety/Depression")
    if result1 == 0:
        column_2.write("Normal")
    
    # if result2 == 1:
    #     column_3.write("Anxiety/Depression")
    # if result2 == 0:
    #     column_3.write("Normal")

user_text = st.text_input("Enter a Sentence")


if user_text is not None:
    cleaned_text = text_cl(user_text)
    if st.button('Detect'):
        Anxiety_detection(cleaned_text)
