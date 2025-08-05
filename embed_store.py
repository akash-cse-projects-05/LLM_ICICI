import uuid
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from extract_chunks import extract_chunks_from_pdf
import os

import os
from dotenv import load_dotenv

load_dotenv()

# Load from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "policy-based"
# Create Pinecone client with env
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer('all-mpnet-base-v2')


def embed_and_store(pdf_path):
    print(f"Processing: {pdf_path}")
    chunks = extract_chunks_from_pdf(pdf_path)
    print(f"Total Chunks Extracted: {len(chunks)}")

    to_upsert = []
    for chunk in chunks:
        vector = model.encode(chunk).tolist()
        vector_id = str(uuid.uuid4())
        metadata = {"text": chunk, "source": pdf_path}
        to_upsert.append((vector_id, vector, metadata))

    print("Uploading to Pinecone...")
    index.upsert(vectors=to_upsert)
    print("✅ Upload complete!")

if __name__ == "__main__":
    pdf_file = os.path.join("data", "ICIHLIP22012V012223.pdf")
    embed_and_store(pdf_file)
