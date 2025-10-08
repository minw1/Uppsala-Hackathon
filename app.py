import os
from typing import List
import streamlit as st
from ask_question import ask_question

st.set_page_config(page_title="Chat Uppsala", page_icon="💬")
st.title("Chat Uppsala")

if not os.environ.get("GOOGLE_API_KEY"):
    st.info("If your backend uses Google APIs, set GOOGLE_API_KEY before running.")


if "messages" not in st.session_state: 
    st.session_state.messages = [ { "role": "assistant", "content": "Hej, I'm your Support Guide Chatbot from the municipality of Uppsala! I can help you "
                                "find information about municipal support and servies - like activities, contact "
                                "persons or applications. Vi kan också chatta på svenska - or any other language!" } ]

# 1) Render existing history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 2) Handle new input (captured this run)
prompt = st.chat_input("Ask your questions here…")
if prompt:
    # Put the user message in a short-lived 'pending' slot so we can show it immediately
    st.session_state["pending_user"] = prompt

# 3) If there is a pending user message, show it immediately and get the answer
if st.session_state.get("pending_user") is not None:
    pending = st.session_state["pending_user"]

    # Show the user's message right away (before RAG finishes)
    with st.chat_message("user"):
        st.markdown(pending)

    # Compute the answer with a spinner; don't write final text yet
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = ask_question(pending)
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"

    # Commit both to history, clear pending, and rerun so they render once via the loop
    st.session_state.messages.append({"role": "user", "content": pending})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state["pending_user"] = None
    st.rerun()
