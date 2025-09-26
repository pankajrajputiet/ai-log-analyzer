import time
from datetime import datetime
from src.ingest import load_logs, chunk_logs, build_faiss_index, load_cached_index
from src.analyze import generate_report
from src.utils import clean_reports, clean_cache

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")

def run(reset_cache=False, reset_reports=False, generate_analysis=True):
    log("🚀 Starting log analysis pipeline...")
    start_total = time.perf_counter()

    if reset_reports:
        clean_reports()
        log("🧹 Cleaned reports")

    if reset_cache:
        clean_cache()
        log("🗑️ Cleaned FAISS cache")

    try:
        index, texts, model = load_cached_index()
        log("📦 Loaded cached FAISS index")
    except Exception as e:
        log(f"⚠️ Failed to load cache: {e}")
        docs = load_logs()
        if not docs:
            log("⚠️ No log files found. Exiting.")
            return
        chunks = chunk_logs(docs)
        index, texts, model = build_faiss_index(chunks)
        log("🔗 Built new FAISS index")

    if generate_analysis:
        success = generate_report(index, texts, model)
        if not success:
            log("⚠️ Report generation skipped due to empty context.")

    log(f"✅ Total pipeline time: {time.perf_counter() - start_total:.2f} sec")

if __name__ == "__main__":
    run()