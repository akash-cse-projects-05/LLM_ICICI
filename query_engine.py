from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# Load from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"

   # Use your index name

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# Load the embedding model (same as embedding phase)
model = SentenceTransformer('all-mpnet-base-v2')


def query_index(query_text, top_k=5):
    # Embed the query
    query_vector = model.encode(query_text).tolist()

    # Query Pinecone for similar vectors
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    # Return matches with metadata
    return result['matches']

if __name__ == "__main__":
    q = "Your question or search text here"
    matches = query_index(q)
    for i, match in enumerate(matches):
        print(f"{i+1}. Score: {match['score']:.4f}")
        print(f"Text: {match['metadata']['text']}\n")
