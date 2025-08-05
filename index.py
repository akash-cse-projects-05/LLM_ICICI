import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Config
PINECONE_API_KEY = "pcsk_48RnJE_G9v28tLRCRYrVNBTUJr5P7ec4iqzrpa1yS2qC8kzRMVhbUBLRphZNmcGMjmFywR"
PINECONE_ENV = "aped-4627-b74a"
INDEX_NAME = "policy-based"  # Replace with your actual index name

# Init Pinecone & model
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-mpnet-base-v2')

st.title("Policy-based Chatbot")

query = st.text_input("Ask a question:")

if query:
    query_vector = model.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=3, include_metadata=True)

    if result['matches']:
        st.write("### Top Answers:")
        for i, match in enumerate(result['matches']):
            text = match['metadata']['text']
            score = match['score']
            st.markdown(f"**{i+1}. (Score: {score:.4f})** {text[:500]}...")
    else:
        st.write("No matching results found.")
