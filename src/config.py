import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_DIR= "embeddings/chroma"
LOG_DIR = "data/logs"
REPORT_DIR = "data/reports"


