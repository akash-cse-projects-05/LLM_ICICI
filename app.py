import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os 
from dotenv import load_dotenv
# === CONFIGURATION ===
load_dotenv()

# Load from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based" # Replace with your actual index name
# === INITIALIZE Pinecone and Sentence Embedding Model ===
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-mpnet-base-v2')  # ✅ Must match index embedding model

# === STREAMLIT INTERFACE ===
st.set_page_config(page_title="Policy Assistant", layout="centered")
st.title("📜 Policy Assistant Chatbot")
st.markdown("Ask any question related to your policy, and we'll help you find the most relevant information!")

# === USER INPUT ===
query = st.text_input("🔎 Type your policy-related question here:")

if query:
    # Encode the query into a vector
    query_vector = model.encode(query).tolist()

    # Query Pinecone index
    result = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    matches = result.get("matches", [])

    if matches:
        st.subheader("✅ Top Results:")
        found = False
        for i, match in enumerate(matches):
            score = match.get("score", 0)
            metadata = match.get("metadata", {})
            text = metadata.get("text", "[No text available]")

            # Display only if score is reasonably high
            if score > 0.7:
                found = True
                st.markdown(f"""
                **{i+1}. Relevance Score: {score:.4f}**
                > {text[:500]}...
                """)
        
        if not found:
            st.warning("⚠️ No highly relevant results found. Try rephrasing your question.")
    else:
        st.error("❌ No results found. Please try again with a different query.")
