import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

#Connecting to vector db

ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name= "compliance_docs", embedding_function = ef)


#Connect to Groq

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

#Ask a question

question= "What are the penalties of violating the GDPR?"

#Retrieve relevant chunks from the vector database

results = collection.query(query_texts=[question], n_results=3)

#Build the context from retieved chunks
context_parts= []
for i,doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i].get("source","unknown")
    page = results["metadatas"][0][i].get("page","?")
    context_parts.append(f"Source: {source}, Page: {page}\n{doc}")

context = "\n".join(context_parts)

#Send the context to the llm with instructions
prompt = f"""You are a compliance assistant. Answer the question using ONLY the context provided below.
Rules:
- If the context does not contain the answer, say "I cannot find this in the provided documents."
- Cite the source document and page number for every claim you make.
- Be precise and direct.

Context:
{context}

Question: {question}

Answer:"""

# Get the response from the LLM
response = llm.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
)

print(f"Question: {question}\n")
print(f"Answer:\n{response.choices[0].message.content}")