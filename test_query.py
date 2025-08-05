from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os 
load_dotenv()

# Load from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"
# Your Pinecone config
  # Replace with your actual index name

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# Load embedding model (must match your index dimension)
model = SentenceTransformer('all-mpnet-base-v2')

def test_query():
    query_text = "What is the policy on data privacy?"
    query_vector = model.encode(query_text).tolist()

    # Query Pinecone
    result = index.query(vector=query_vector, top_k=3, include_metadata=True)

    print(f"Top results for query: '{query_text}'\n")
    for i, match in enumerate(result['matches']):
        text = match['metadata']['text']
        snippet = text if len(text) <= 300 else text[:300] + "..."
        print(f"{i+1}. Score: {match['score']:.4f}")
        print(f"Text snippet: {snippet}\n")

if __name__ == "__main__":
    test_query()
