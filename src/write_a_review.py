# from fastai import *
import streamlit as st
from fastai.text.all import *
from fastai.learner import load_learner

model_here = load_learner("./imdb1.pkl")

TEXT = "I am gonna start BTC mining,"

N_WORDS = 40
N_SENTENCES = 2
# preds = [
#     model_here.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)
# ]

# print("\n".join(preds))


def predict_next(text):
    preds = [
        model_here.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)
    ]
    return "\n".join(preds)


text = st.text_area("enter your review here")

if st.button("write more"):
    response = predict_next(text)
    "full review: ", response
