
import os
import json
import yaml
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

def extract_structural():
    print("Extracting structural edges from taxonomy...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    edges = []
    
    # helper for json parsing
    def parse_json_field(val):
        if not val: return []
        if isinstance(val, list): return val
        if isinstance(val, str):
            try:
                loaded = json.loads(val)
                if isinstance(loaded, list): return loaded
                return [loaded]
            except:
                return [val] # assume raw string UUID if not json
        return []

    try:
        with conn.cursor() as cur:
            # Check concept columns
            cur.execute(f"""
                SELECT concept_id, facet_ids, discipline_ids, parent_concept_ids, related 
                FROM {CONFIG['db']['concepts_table']}
            """)
            
            rows = cur.fetchall()
            print(f"Analyzing {len(rows)} concepts for structure...")
            
            for row in rows:
                cid = str(row[0])
                facet_ids = parse_json_field(row[1])
                disc_ids = parse_json_field(row[2])
                parent_ids = parse_json_field(row[3])
                related = parse_json_field(row[4])
                
                # HAS_FACET
                for fid in facet_ids:
                    edges.append((cid, fid, "HAS_FACET", 1.0, 0))
                
                # IN_DISCIPLINE
                for did in disc_ids:
                    # check if did is dict? (sometimes related is dict)
                    # discipline_ids usually list of strings from other scripts?
                    # extract_physh_taxonomy.py: discipline_ids: set
                    if isinstance(did, dict): did = did.get("id")
                    if did:
                        edges.append((cid, did, "IN_DISCIPLINE", 1.0, 0))
                        
                # PARENT_OF (Parent -> Child(Current))
                for pid in parent_ids:
                    if isinstance(pid, dict): pid = pid.get("id")
                    if pid:
                        edges.append((pid, cid, "PARENT_OF", 1.0, 0))
                
                # RELATED_TO (Symmetric usually, but we insert directed)
                for r in related:
                    # related is often list of objects in PhySH: [{"concept_id": ...}]
                    rid = None
                    if isinstance(r, dict):
                        rid = r.get("concept_id") or r.get("id")
                    elif isinstance(r, str):
                        rid = r
                    
                    if rid:
                        edges.append((cid, rid, "RELATED_TO", 1.0, 0))

        # Insert
        print(f"Found {len(edges)} structural edges. Inserting...")
        with conn.cursor() as cur:
            args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in edges)
            # Use ON CONFLICT DO NOTHING to avoid duplicates if re-run
            # Warning: Mogrify with huge list might hit memory limits. Batch it.
            
            batch_size = 5000
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i+batch_size]
                args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in batch)
                cur.execute(f"""
                    INSERT INTO third_party_aps.ontology_edges (source_id, target_id, relation, weight, support) 
                    VALUES {args_str}
                    ON CONFLICT (source_id, target_id, relation) DO UPDATE 
                    SET weight = EXCLUDED.weight
                """)
                print(f"Inserted batch {i} to {i+batch_size}...", end='\r')
            
            conn.commit()
            print("\nStructural edges inserted.")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    extract_structural()
