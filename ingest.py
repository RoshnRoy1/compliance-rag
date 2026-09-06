import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Step 1: Load all PDFs from docs folder
docs_folder ="docs"
all_pages =[]

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(docs_folder, filename)
        loader = PyPDFLoader(path)
        print(f"Loading {filename}...")
        loader = PyPDFLoader(path)
        pages = loader.load()
        print(f" -> {len(pages)} pages") 
        all_pages.extend(pages)

print(f"\nTotal pages loaded: {len(all_pages)}")

#Step 2: Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n",". ",", ", " ", ""]
)

chunks = splitter.split_documents(all_pages) #splitter is a 
print(f"\nTotal chunks created: {len(chunks)}")

#Step 3: Inspect a few chunks so you see what you're working with

print("\n Example chunks ")
print(f"Source: {chunks[0].metadata['source']}")
print(f"Page: {chunks[0].metadata['page']}")
print(f"Length: {len(chunks[0].page_content)} chars")
print(f"Content: {chunks[0].page_content[:300]}...")  # Print first 300 characters

#Step 4 : Store chunks in a vec db

import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.DefaultEmbeddingFunction()
client =chromadb.PersistentClient(path="./chroma_db") #this creates a database that saves to a folder called chroma_db on the disk

#Delete old collection if it exists
try:
    client.delete_collection("compliance_docs") #if you've run this script before, it wipes the old data so you start fresh.
    print("Deleted old collection 'compliance_docs'")
except:
    pass

# Create a new collection
collection = client.create_collection(name="compliance_docs", embedding_function=ef) 
#creates a collection (like a table) called "compliance_docs." 
#Every document added to it will automatically be converted to a 384-number embedding by the same model you used before

#Add chunks in batches

"""
In your test script you added 5 documents in one call. Chroma can't handle 5,636 at once, so you feed them in groups of 100. 
The loop says: take chunks 0-99, embed and store them. Then 100-199.
 Then 200-299. And so on until all 5,636 are stored. It prints progress so you know it's working and hasn't frozen.

For each batch, three things get stored together:
 the documents (the actual text, so you can read it later), the metadatas (which PDF and which page the chunk came from, this is what powers your citations),
  and the ids (a unique label like chunk_0, chunk_1 so you can reference them).
"""

batch_size = 100
for i in range(0,len(chunks), batch_size):
    batch = chunks[i:i+ batch_size]
    collection.add(
        documents=[c.page_content for c in batch],
        metadatas= [c.metadata for c in batch],
        ids=[f"chunk_{i+j}" for j,c in enumerate(batch)],
        )
    print(f" Embedded batch {i//batch_size + 1}/ {len(chunks) // batch_size + 1 }")

print(f"\n DOne! {collection.count()} chunks stored in vec db!")
