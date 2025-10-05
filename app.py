import os
from typing import List

import streamlit as st
from ask_question import ask_question 

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("RAG Chatbot")

# Optional heads-up if keys are needed by your backend
if not os.environ.get("GOOGLE_API_KEY"):
    st.info("If your backend uses Google APIs, set GOOGLE_API_KEY before running.")

# Minimal in-memory chat history
if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User input
prompt = st.chat_input("Ask about your documents…")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = ask_question(prompt)
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("Frontend only · talks to your RAG backend via answer_question()")
