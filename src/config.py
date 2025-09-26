# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")

# Directory paths
LOG_DIR = "data/logs"
REPORT_DIR = "data/reports"
CACHE_DIR = "data/cache"

# Ensure directories exist
for path in [LOG_DIR, REPORT_DIR, CACHE_DIR]:
    os.makedirs(path, exist_ok=True)

# Embedding model and retrieval settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 3