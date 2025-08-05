from pinecone import Pinecone, ServerlessSpec

# ✅ Replace with your actual API key or use environment variables
PINECONE_API_KEY = "pcsk_48RnJE_G9v28tLRCRYrVNBTUJr5P7ec4iqzrpa1yS2qC8kzRMVhbUBLRphZNmcGMjmFywR"
INDEX_NAME = "policy-index"

# ✅ Dimension must match your embedding model
DIMENSION = 384  # all-MiniLM-L6-v2

# ✅ Use correct Pinecone region for serverless
CLOUD = "aws"
REGION = "us-east-1"  # or whatever region you selected in Pinecone console

# ✅ Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )
    print(f"✅ Created index: {INDEX_NAME}")
else:
    print(f"ℹ️ Index '{INDEX_NAME}' already exists.")
