"""Collect embeddings for cities using OpenAI Batch API.

This script:
1. Creates a batch job with all cities that don't have embeddings yet
2. Polls for completion
3. Stores results in SQLite database

Can be stopped and restarted - only processes cities without embeddings.
"""

import csv
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm


MODEL = "text-embedding-3-large"
DB_PATH = Path(__file__).parent.parent / "data" / "cities.db"
CITIES_CSV = Path(__file__).parent.parent / "data" / "cities.csv"
BATCH_INPUT_FILE = Path(__file__).parent.parent / "data" / "batch_input.jsonl"
BATCH_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "batch_output.jsonl"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with cities table."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            region TEXT,
            country TEXT NOT NULL,
            embedding BLOB,
            UNIQUE(city, region, country)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batch_jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            completed_at TEXT
        )
    """)
    conn.commit()
    return conn


def load_cities_to_db(conn: sqlite3.Connection, csv_path: Path) -> None:
    """Load cities from CSV into database if not already present."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cities = list(reader)
    
    cursor = conn.cursor()
    inserted = 0
    for city in tqdm(cities, desc="Loading cities to DB"):
        try:
            cursor.execute(
                "INSERT OR IGNORE INTO cities (city, region, country) VALUES (?, ?, ?)",
                (city["city"], city["region"], city["country"])
            )
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    print(f"Inserted {inserted} new cities, {len(cities) - inserted} already existed")


def get_cities_without_embeddings(conn: sqlite3.Connection) -> list[tuple[int, str, str, str]]:
    """Get cities that don't have embeddings yet."""
    cursor = conn.execute(
        "SELECT id, city, region, country FROM cities WHERE embedding IS NULL"
    )
    return cursor.fetchall()


def make_prompt(city: str, region: str, country: str) -> str:
    """Create the embedding prompt for a city."""
    if region:
        return f"Is {city}, {region}, {country} in continental Europe?"
    return f"Is {city}, {country} in continental Europe?"


def create_batch_input(cities: list[tuple[int, str, str, str]], output_path: Path) -> None:
    """Create JSONL file for batch API input."""
    with open(output_path, "w", encoding="utf-8") as f:
        for city_id, city, region, country in tqdm(cities, desc="Creating batch input"):
            prompt = make_prompt(city, region, country)
            request = {
                "custom_id": str(city_id),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": MODEL,
                    "input": prompt,
                }
            }
            f.write(json.dumps(request) + "\n")
    print(f"Created batch input with {len(cities)} requests")


def submit_batch_job(client: OpenAI, input_path: Path) -> str:
    """Submit batch job and return batch ID."""
    # Upload input file
    print("Uploading batch input file...")
    with open(input_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    
    print(f"Uploaded file: {batch_file.id}")
    
    # Create batch job
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    
    print(f"Created batch job: {batch.id}")
    return batch.id


def wait_for_batch(client: OpenAI, batch_id: str) -> dict:
    """Poll for batch completion with progress bar."""
    print(f"Waiting for batch {batch_id} to complete...")
    
    with tqdm(desc="Batch progress", unit="%", total=100) as pbar:
        last_progress = 0
        while True:
            batch = client.batches.retrieve(batch_id)
            
            if batch.status == "completed":
                pbar.update(100 - last_progress)
                print(f"\nBatch completed! Processed {batch.request_counts.completed} requests")
                return batch
            elif batch.status == "failed":
                raise RuntimeError(f"Batch failed: {batch.errors}")
            elif batch.status == "expired":
                raise RuntimeError("Batch expired")
            elif batch.status == "cancelled":
                raise RuntimeError("Batch was cancelled")
            
            # Update progress based on completed requests
            if batch.request_counts.total > 0:
                progress = int(100 * batch.request_counts.completed / batch.request_counts.total)
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
            
            time.sleep(10)


def download_and_store_results(
    client: OpenAI, 
    batch: dict, 
    conn: sqlite3.Connection,
    output_path: Path
) -> None:
    """Download batch results and store embeddings in database."""
    # Download output file
    print("Downloading results...")
    output_file_id = batch.output_file_id
    content = client.files.content(output_file_id)
    
    with open(output_path, "wb") as f:
        f.write(content.content)
    
    print(f"Downloaded results to {output_path}")
    
    # Parse and store embeddings
    print("Storing embeddings in database...")
    cursor = conn.cursor()
    
    with open(output_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    stored = 0
    errors = 0
    for line in tqdm(lines, desc="Storing embeddings"):
        result = json.loads(line)
        city_id = int(result["custom_id"])
        
        if result.get("error"):
            errors += 1
            print(f"Error for city {city_id}: {result['error']}")
            continue
        
        # Extract embedding from response
        response = result["response"]
        if response["status_code"] == 200:
            embedding = response["body"]["data"][0]["embedding"]
            # Store as binary blob (more efficient)
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute(
                "UPDATE cities SET embedding = ? WHERE id = ?",
                (embedding_blob, city_id)
            )
            stored += 1
        else:
            errors += 1
            print(f"API error for city {city_id}: {response}")
    
    conn.commit()
    print(f"Stored {stored} embeddings, {errors} errors")


def check_pending_batch(conn: sqlite3.Connection, client: OpenAI) -> str | None:
    """Check if there's a pending batch job to resume."""
    cursor = conn.execute(
        "SELECT id FROM batch_jobs WHERE status NOT IN ('completed', 'failed', 'expired', 'cancelled') ORDER BY created_at DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row:
        batch_id = row[0]
        # Verify it still exists
        try:
            batch = client.batches.retrieve(batch_id)
            if batch.status in ("validating", "in_progress", "finalizing"):
                return batch_id
        except Exception:
            pass
    return None


def save_batch_job(conn: sqlite3.Connection, batch_id: str, status: str) -> None:
    """Save batch job status to database."""
    conn.execute(
        "INSERT OR REPLACE INTO batch_jobs (id, status, created_at) VALUES (?, ?, datetime('now'))",
        (batch_id, status)
    )
    conn.commit()


def main() -> None:
    """Main entry point."""
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    # Initialize database
    print(f"Using database: {DB_PATH}")
    conn = init_db(DB_PATH)
    
    # Load cities from CSV
    load_cities_to_db(conn, CITIES_CSV)
    
    # Check for pending batch job
    pending_batch_id = check_pending_batch(conn, client)
    if pending_batch_id:
        print(f"Found pending batch job: {pending_batch_id}")
        batch = wait_for_batch(client, pending_batch_id)
        download_and_store_results(client, batch, conn, BATCH_OUTPUT_FILE)
        save_batch_job(conn, pending_batch_id, "completed")
    
    # Get cities without embeddings
    cities = get_cities_without_embeddings(conn)
    
    if not cities:
        print("All cities have embeddings!")
        return
    
    print(f"Found {len(cities)} cities without embeddings")
    
    # Create batch input
    create_batch_input(cities, BATCH_INPUT_FILE)
    
    # Submit batch job
    batch_id = submit_batch_job(client, BATCH_INPUT_FILE)
    save_batch_job(conn, batch_id, "submitted")
    
    # Wait for completion
    batch = wait_for_batch(client, batch_id)
    
    # Download and store results
    download_and_store_results(client, batch, conn, BATCH_OUTPUT_FILE)
    save_batch_job(conn, batch_id, "completed")
    
    # Final stats
    cursor = conn.execute("SELECT COUNT(*) FROM cities WHERE embedding IS NOT NULL")
    total_with_embeddings = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM cities")
    total = cursor.fetchone()[0]
    print(f"\nDone! {total_with_embeddings}/{total} cities have embeddings")
    
    conn.close()


if __name__ == "__main__":
    main()
