# import getpass
# import os
# import time
# import numpy as np
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from uuid import uuid4
# from langchain_core.documents import Document
# from langchain_ollama import OllamaEmbeddings  # Custom module for Ollama embeddings (replace if necessary)

# # Prompt for API key if not set
# if not os.getenv("PINECONE_API_KEY"):
#     os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

# pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)

# # List existing indexes
# print(pc.list_indexes())

# # Define the index name
# index_name = "medium"  # Use your existing index
# embedding_dimension = 4096  # Replace with the Ollama embedding dimension

# # Ensure the index exists (create if not already present)
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if index_name not in existing_indexes:
#     pc.create_index(
#         name=index_name,
#         dimension=embedding_dimension,  # Set to match Ollama embedding dimension
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     while not pc.describe_index(index_name).status["ready"]:
#         time.sleep(1)

# # Connect to the index
# index = pc.Index(index_name)

# # Initialize Ollama embeddings
# ollama_model_name = "llama3"  # Replace with your Ollama model name
# embeddings_model = OllamaEmbeddings(model=ollama_model_name)

# # Read and process the `mediumblog.txt` file
# file_path = os.path.abspath("mediumblog1.txt")  # Ensure this file exists in the same directory or provide the full path
# documents = []
# with open(file_path, "r", encoding="utf-8") as f:
#     for line_num, line in enumerate(f, start=1):
#         content = line.strip()
#         if content:  # Skip empty lines
#             documents.append(
#                 Document(
#                     page_content=content,
#                     metadata={"source": "mediumblog", "line_number": line_num},
#                 )
#             )

# # Generate unique IDs for documents
# uuids = [str(uuid4()) for _ in documents]

# # Add documents to the vector store
# for doc, uuid in zip(documents, uuids):
#     embedding = embeddings_model.embed_query(doc.page_content)  # Generate embedding
#     embedding = np.array(embedding, dtype=np.float32)  # Ensure float32
#     index.upsert(vectors=[(uuid, embedding, {"text": doc.page_content, **doc.metadata})])

# print("Documents from mediumblog.txt successfully added to Pinecone.")

# # Perform similarity search on the vector store
# vector_store = PineconeVectorStore(index=index, embedding=embeddings_model)

# query = "Explain the core concepts of LangChain for AI applications."  # Example query
# results = vector_store.similarity_search(
#     query,
#     k=3,
#     filter={"source": "mediumblog"},
# )

# # Print results
# print("Search Results:")
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")


import os
import getpass
import time
from uuid import uuid4
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

# Prompt for API keys if not set
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define the index name
index_name = "embedding-ollama"  # Change if desired
embedding_dimension = 4096  # Adjust to match Ollama embeddings size

# Check and create index if not exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,  # Ensure this matches Ollama embeddings size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)

# Initialize embeddings model using Ollama
embeddings_model = OllamaEmbeddings(model='llama3')

# Load text from file
file_path = os.path.abspath("error2.txt")
print(f"Loading document from: {file_path}")
with open(file_path, "r", encoding="utf-8") as file:
    medium_blog_content = file.read()

# Document splitting
print("Splitting...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Create a document and split into chunks
document = [Document(page_content=medium_blog_content, metadata={"source": "medium-blog"})]
texts = text_splitter.split_documents(document)
print(f"Created {len(texts)} chunks.")

# Generate unique IDs for each chunk
uuids = [str(uuid4()) for _ in texts]

# Ingest into Pinecone
print("Ingesting...")
for chunk, uuid in zip(texts, uuids):
    embedding = embeddings_model.embed_query(chunk.page_content)  # Generate embedding
    embedding = np.array(embedding, dtype=np.float32)  # Ensure float32
    index.upsert(vectors=[(uuid, embedding, {"text": chunk.page_content, **chunk.metadata})])

print("Finish")
