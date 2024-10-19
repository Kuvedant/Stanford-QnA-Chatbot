# app.py

import streamlit as st
import requests

st.title("SQuADBot")

if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

def get_answer(question):
    response = requests.post("http://localhost:8000/ask", json={"question": question})
    return response.json()['answer']

user_input = st.text_input("You:", key="input")

if st.button("Send"):
    answer = get_answer(user_input)
    st.session_state['conversation'].append({'user': user_input, 'bot': answer})

for turn in st.session_state['conversation']:
    st.write(f"You: {turn['user']}")
    st.write(f"Bot: {turn['bot']}")
