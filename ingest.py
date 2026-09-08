import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from config import DOCS_FOLDER, CHROMA_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

# Step 1: Load PDFs
all_pages = []
for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(DOCS_FOLDER, filename)
        print(f"Loading {filename}...")
        loader = PyPDFLoader(path)
        pages = loader.load()
        print(f"  -> {len(pages)} pages")
        all_pages.extend(pages)

print(f"\nTotal pages loaded: {len(all_pages)}")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(all_pages)
print(f"Total chunks created: {len(chunks)}")

# Step 3: Embed and store
ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

batch_size = 100
total_batches = (len(chunks) // batch_size) + 1
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    collection.add(
        documents=[c.page_content for c in batch],
        metadatas=[c.metadata for c in batch],
        ids=[f"chunk_{i + j}" for j, c in enumerate(batch)],
    )
    print(f"  Embedded batch {i // batch_size + 1}/{total_batches}")

print(f"\nDone! {collection.count()} chunks stored.")