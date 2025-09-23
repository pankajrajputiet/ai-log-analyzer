import shutil
import os
from src.config import REPORT_DIR, CACHE_DIR

def clean_reports():
    if os.path.exists(REPORT_DIR):
        shutil.rmtree(REPORT_DIR)

def clean_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)