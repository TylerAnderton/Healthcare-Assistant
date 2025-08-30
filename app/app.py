import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺", layout="wide")

st.sidebar.title("Settings")
model_provider = os.getenv("MODEL_PROVIDER", "ollama")
st.sidebar.markdown(f"**Model provider:** {model_provider}")
st.sidebar.markdown("This tool is not medical advice. Consult a clinician.")

st.title("🩺 Healthcare Assistant (Local)")
st.caption("Ask questions about your labs, medications, and WHOOP data. Sources are cited when available.")

if "messages" not in st.session_state:
    st.session_state.messages = []

from chains.chat import answer_question, has_vectorstore

if not has_vectorstore():
    st.warning("No vector index found. Please run ingestion and indexing steps (see README). The chatbot will still respond, but without document retrieval.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your question (e.g., 'Show trends for my thyroid labs in the last year')")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass prior turns (excluding the just-appended user message)
            history = st.session_state.messages[:-1]
            ans, sources = answer_question(prompt, history=history)
            st.markdown(ans)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")
    st.session_state.messages.append({"role": "assistant", "content": ans})
