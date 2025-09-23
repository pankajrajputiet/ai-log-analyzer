import time
from src.ingest import load_logs, chunk_logs, build_faiss_index, load_cached_index
from src.analyze import generate_report
from src.utils import clean_reports, clean_cache

def run(reset_cache=False, reset_reports=False, generate_analysis=True):
    print("🚀 Starting log analysis pipeline...")
    start_total = time.perf_counter()

    if reset_reports:
        clean_reports()
        print("🧹 Cleaned reports")

    if reset_cache:
        clean_cache()
        print("🗑️ Cleaned FAISS cache")

    try:
        index, texts, model = load_cached_index()
        print("📦 Loaded cached FAISS index")
    except:
        docs = load_logs()
        if not docs:
            print("⚠️ No log files found. Exiting.")
            return
        chunks = chunk_logs(docs)
        index, texts, model = build_faiss_index(chunks)
        print("🔗 Built new FAISS index")

    if generate_analysis:
        generate_report(index, texts, model)

    print(f"✅ Total pipeline time: {time.perf_counter() - start_total:.2f} sec")

if __name__ == "__main__":
    run()