import chromadb
from chromadb.utils import embedding_functions

#Load the saved db

ef= embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name= "compliance_docs", embedding_function = ef)

print(f"Collection '{collection.count()}' chunks loaded.")

#Ask a question

question = "When must a data breach be reported?"
question = "Within how many hours must a personal data breach be notified to the supervisory authority under GDPR?"

results = collection.query(query_texts= [question], n_results=3)

print(f"\nQuestion: {question}\n")
print("The Top 3 Results are:")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    source = meta.get('source',"unknown")
    page = meta.get('page',"?")
    print(f"\n --- Result{i+1} (distance: {dist:.4f}) ---")
    print(f"Source: {source}, Page: {page}")
    print(f"Document: {doc} \n")