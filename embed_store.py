import uuid
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from extract_chunks import extract_chunks_from_pdf
import os

PINECONE_API_KEY = "pcsk_48RnJE_G9v28tLRCRYrVNBTUJr5P7ec4iqzrpa1yS2qC8kzRMVhbUBLRphZNmcGMjmFywR"
PINECONE_ENV = "aped-4627-b74a"
INDEX_NAME = "policy-based"  # Match your Pinecone index name exactly

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
