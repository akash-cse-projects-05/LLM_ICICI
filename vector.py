import pinecone
from dotenv import load_dotenv
import os 
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Check if index exists; if not, create it
if "policy-index" not in pinecone.list_indexes():
    pinecone.create_index(
        name="policy-index",
        dimension=768,  
        metric="cosine"
    )
print("Index ready!")
