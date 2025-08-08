


import os
import requests
import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "policy-based"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer('all-mpnet-base-v2')


def query_pinecone(query_text, top_k=3):
    query_vector = model.encode(query_text).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result.get('matches', [])


def generate_answer_groq(context, question):
    api_url = "https://api.groq.com/openai/v1/completions"  # Updated Groq endpoint

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""You are a helpful assistant. Use the following policy information to answer the question:

Context:
{context}

Question: {question}
Answer concisely and clearly:"""

    payload = {
        "model": "gpt-4o-mini",
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.3
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        resp_json = response.json()
        return resp_json.get("choices", [{}])[0].get("text", "").strip()
    except requests.exceptions.RequestException as e:
        return f"⚠️ Unable to reach Groq API: {e}\n\nHere’s some relevant info:\n\n" + context[:500] + "..."


# --------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Policy Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Policy Chatbot ")
st.write("Ask questions about your policy documents. Type below and get instant answers.")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your Question:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Query Pinecone
    matches = query_pinecone(user_input)
    if matches:
        context = "\n\n".join([match['metadata'].get('text', '') for match in matches])
        answer = generate_answer_groq(context, user_input)
    else:
        answer = "Sorry, no relevant information found."

    # Add bot message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")
