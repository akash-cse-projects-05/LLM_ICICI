from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import re
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load env variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"

# Initialize Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-mpnet-base-v2')

app = Flask(__name__)

def clean_text(text):
    """
    Clean the retrieved text by removing IDs, unwanted headers,
    multiple spaces, and newlines.
    """
    # Remove CIN, UIN policy numbers/IDs
    text = re.sub(r'(CIN|UIN)\s*:\s*\S+', '', text)
    # Remove lines starting with digits and 'Golden Shield'
    text = re.sub(r'^\d+\.\s*Golden Shield.*', '', text, flags=re.MULTILINE)
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({'response': "Please enter a question."})

    logging.info(f"Received query: {user_input}")

    # Embed the query and search Pinecone index
    query_vector = model.encode(user_input).tolist()
    result = index.query(vector=query_vector, top_k=3, include_metadata=True)
    matches = result.get('matches', [])

    if not matches:
        logging.info("No relevant matches found.")
        return jsonify({'response': "Sorry, no relevant info found."})

    # Combine and clean all matched texts
    combined_text = " ".join(clean_text(match['metadata'].get('text', '')) for match in matches)
    logging.info(f"Combined raw text length: {len(combined_text)}")

    # Limit response length intelligently to ~600 chars without cutting words
    max_len = 600
    if len(combined_text) > max_len:
        combined_text = combined_text[:max_len].rsplit(' ', 1)[0] + "..."

    logging.info(f"Returning response of length: {len(combined_text)}")

    return jsonify({'response': combined_text})

if __name__ == "__main__":
    app.run(debug=True)
