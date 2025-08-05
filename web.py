from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import re

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"

# Initialize Pinecone client and index (assuming pinecone-client v2)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer('all-mpnet-base-v2')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def clean_text(text):
    # Remove policy numbers and IDs like CIN, UIN
    text = re.sub(r'(CIN|UIN)\s*:\s*\S+', '', text)
    # Remove repeated headers like "Golden Shield 16" or lines starting with digits
    text = re.sub(r'^\d+\.\s*Golden Shield.*', '', text, flags=re.MULTILINE)
    # Replace multiple newlines and tabs with space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({'response': "Please enter a question."})

    query_vector = model.encode(user_input).tolist()
    result = index.query(vector=query_vector, top_k=3, include_metadata=True)
    matches = result.get('matches', [])

    if not matches:
        return jsonify({'response': "Sorry, no relevant info found."})

    combined_text = ""
    for match in matches:
        raw_text = match['metadata'].get('text', '')
        cleaned = clean_text(raw_text)
        combined_text += cleaned + " "

    combined_text = combined_text.strip()
    # Truncate to 600 chars for readability
    if len(combined_text) > 600:
        # Avoid cutting mid-word
        combined_text = combined_text[:600].rsplit(' ', 1)[0] + "..."

    return jsonify({'response': combined_text})
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_input = data.get('message', '')

#     if not user_input:
#         return jsonify({'response': "Please enter a question."})

#     # Embed and query Pinecone
#     query_vector = model.encode(user_input).tolist()
#     result = index.query(vector=query_vector, top_k=3, include_metadata=True)
#     matches = result.get('matches', [])

#     if not matches:
#         return jsonify({'response': "Sorry, no relevant info found."})

#     # Build response from top matches
#     response_text = ""
#     for i, match in enumerate(matches):
#         text = match['metadata'].get('text', 'No text available')
#         snippet = text if len(text) <= 300 else text[:300] + "..."
#         response_text += f"<strong>{i+1}.</strong> {snippet}<br><br>"

#     return jsonify({'response': response_text})

if __name__ == "__main__":
    app.run(debug=True)
