"""Migrate database embeddings from JSON text to binary numpy arrays.
This significantly reduces the database size.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path('data/cities.db')

def migrate():
    print(f"Migrating {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Read all data
    cursor = conn.execute("SELECT id, city, embedding FROM cities WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows with embeddings.")
    
    updates = []
    skipped = 0
    converted = 0
    
    for row in tqdm(rows, desc="Processing"):
        city_id, city, blob = row
        
        # Check if already binary (heuristic: if it's text json, it starts with [)
        try:
            # Try parsing as JSON first (Current format)
            text = blob.decode('utf-8')
            if text.strip().startswith('['):
                # It is JSON
                embedding_list = json.loads(text)
                # Convert to binary
                arr = np.array(embedding_list, dtype=np.float32)
                binary = arr.tobytes()
                updates.append((binary, city_id))
                converted += 1
            else:
                # Might already be binary? 
                # If we assume 1536 dims float32 = 6144 bytes
                if len(blob) == 6144:
                    skipped += 1
                else:
                    print(f"Warning: Unknown format for city {city} (len {len(blob)})")
        except UnicodeDecodeError:
            # It's likely already binary
            skipped += 1
            
    print(f"Ready to update {converted} rows. {skipped} rows seem already binary.")
    
    if updates:
        conn.executemany("UPDATE cities SET embedding = ? WHERE id = ?", updates)
        conn.commit()
        print("Updates committed.")
        
    print("Vacuuming database...")
    conn.execute("VACUUM")
    print("Database vacuumed.")
    conn.close()

if __name__ == '__main__':
    migrate()
