import shutil
import os
from src.config import REPORT_DIR, CACHE_DIR

def clean_reports():
    if os.path.exists(REPORT_DIR):
        shutil.rmtree(REPORT_DIR)
        print(f"Deleted report directory: {REPORT_DIR}")
    else:
        print(f"Report directory not found: {REPORT_DIR}")

def clean_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Deleted cache directory: {CACHE_DIR}")
    else:
        print(f"Cache directory not found: {CACHE_DIR}")