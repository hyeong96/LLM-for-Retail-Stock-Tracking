import streamlit as st
from langchain_util import setup

st.title("LLM-based Merchandise StockTracking: Q&A ")

question = st.text_input("Question: ")

if question:
    chain = setup(verbose=True)
    answer = chain(question)['result']
    st.header("Answer: ")
    st.write(answer)