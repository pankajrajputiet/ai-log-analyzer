from src.ingest import load_logs, chunk_and_store
from src.analyze import generate_report
from src.utils import clean_reports, clean_chroma


def run():
    """
    Main pipeline for log analysis:
    1. Clean old reports and Chroma database.
    2. Load logs from disk.
    3. Split logs into chunks and store them in Chroma DB.
    4. Generate a fresh error analysis report.
    """
    
    print("🚀 Starting log analysis pipeline...")

    # Step 1: Clean previous outputs
    print("🧹 Cleaning previous reports...")
    clean_reports()

    print("🗑️ Cleaning Chroma database...")
    clean_chroma()

    # Step 2: Load log files
    print("📂 Loading log files...")
    docs = load_logs()

    # Step 3: Process logs into chunks and store in vector DB
    print("🔍 Chunking logs and storing into Chroma DB...")
    chunk_and_store(docs)

    # Step 4: Run analysis and generate a report
    print("📝 Generating error report...")
    generate_report()

    print("✅ Report generated successfully!")


if __name__ == "__main__":
    run()
