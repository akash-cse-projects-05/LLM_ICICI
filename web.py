from flask import Flask, render_template, request, jsonify
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import re
import requests
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Load env variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "policy-based"

# Initialize Pinecone and model
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-mpnet-base-v2')

app = Flask(__name__)

def clean_text(text):
    """Clean text by removing IDs, headers, and excess whitespace."""
    text = re.sub(r'(CIN|UIN)\s*:\s*\S+', '', text)
    text = re.sub(r'^\d+\.\s*Golden Shield.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def query_pinecone(query_text, top_k=3):
    """Get top-k relevant matches from Pinecone."""
    query_vector = model.encode(query_text).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result.get('matches', [])

def generate_answer_groq(context, question):
    """Call Groq API to get answer based on context and question."""
    api_url = "https://api.groq.ai/v1/llm/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"""You are a helpful assistant. Use the following policy information to answer the question concisely and clearly.

Context:
{context}

Question: {question}
Answer:"""

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
        logging.error(f"Groq API error: {e}")
        return "⚠️ Unable to reach Groq API. Here's the relevant info:\n\n" + context[:500] + "..."

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

    matches = query_pinecone(user_input)
    if not matches:
        logging.info("No relevant matches found.")
        return jsonify({'response': "Sorry, no relevant info found."})

    # Combine and clean matched texts for context
    combined_context = "\n\n".join(clean_text(match['metadata'].get('text', '')) for match in matches)
    logging.info(f"Context length: {len(combined_context)}")

    # Call Groq LLM API with combined context and question
    answer = generate_answer_groq(combined_context, user_input)
    logging.info("Returning answer from Groq LLM.")

    return jsonify({'response': answer})

if __name__ == "__main__":
    app.run(debug=True)
