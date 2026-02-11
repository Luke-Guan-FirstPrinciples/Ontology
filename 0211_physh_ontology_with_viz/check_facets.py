
import os
import psycopg2
import json
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

def main():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("SELECT to_regclass('third_party_aps.physh_concepts');")
        if not cur.fetchone()[0]:
            print("Table third_party_aps.physh_concepts does not exist.")
            return

        cur.execute("SELECT DISTINCT facet_labels FROM third_party_aps.physh_concepts;")
        rows = cur.fetchall()
        
        unique_labels = set()
        for row in rows:
            # Row matches are likely lists in JSONB or strings if stored as text
            # The previous script implies it stores JSON strings provided by python `json.dumps()` in CSV, 
            # but if it was uploaded to DB, it might be JSONB or Text array.
            # Let's inspect raw.
            raw = row[0]
            if isinstance(raw, list):
                for x in raw: unique_labels.add(x)
            elif isinstance(raw, str):
                try:
                    loaded = json.loads(raw)
                    if isinstance(loaded, list):
                        for x in loaded: unique_labels.add(x)
                    else:
                        unique_labels.add(raw)
                except:
                    unique_labels.add(raw)
        
        print("Distinct Facet Labels found:")
        for l in sorted(unique_labels):
            print(f"- {l}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()
