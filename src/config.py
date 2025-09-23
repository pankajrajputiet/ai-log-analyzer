import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LOG_DIR = "data/logs"
REPORT_DIR = "data/reports"
CACHE_DIR = "data/cache"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 3
