import os
import requests
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Add your Groq API key here
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
    api_url = "https://api.groq.ai/v1/llm/completions"  # Verify this URL with Groq docs

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
        "model": "gpt-4o-mini",  # Check your available model name on Groq
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
        print(f"Error contacting Groq API: {e}")
        # Fallback: return the raw context (or you can return a message)
        return "⚠️ Unable to reach Groq API. Here's the relevant info:\n\n" + context[:500] + "..."

def chatbot():
    print("Welcome to the Policy Chatbot (Groq LLM + Pinecone)!")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        matches = query_pinecone(user_input)
        if not matches:
            print("Bot: Sorry, no relevant info found.")
            continue

        context = "\n\n".join([match['metadata'].get('text', '') for match in matches])

        answer = generate_answer_groq(context, user_input)

        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    chatbot()
