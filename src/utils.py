import shutil
from src.config import REPORT_DIR, CHROMA_DB_DIR

def clean_reports():
    shutil.rmtree(REPORT_DIR, ignore_errors=True)

def clean_chroma():
    shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
