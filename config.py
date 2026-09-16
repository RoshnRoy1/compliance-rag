import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "qwen/qwen3.8-27b"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 500

# Vector database
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "compliance_docs"

# Retrieval
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3

# Documents
DOCS_FOLDER = "docs"

import os
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"