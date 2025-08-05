from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os 
load_dotenv()

# Load from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# Load embedding model (must match your index dimension)
model = SentenceTransformer('all-mpnet-base-v2')

def query_pinecone(query_text, top_k=3):
    query_vector = model.encode(query_text).tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return result['matches']

def chatbot():
    print("Welcome to the Policy Chatbot! Ask anything or type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        matches = query_pinecone(user_input)
        if not matches:
            print("Bot: Sorry, no relevant info found.")
            continue
        print("Bot: Here are the top answers:")
        for i, match in enumerate(matches):
            text = match['metadata']['text']
            snippet = text if len(text) <= 300 else text[:300] + "..."
            print(f"{i+1}. (Score: {match['score']:.4f}) {snippet}")
        print()

if __name__ == "__main__":
    chatbot()
