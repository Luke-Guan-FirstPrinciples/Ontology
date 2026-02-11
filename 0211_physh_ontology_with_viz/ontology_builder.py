
import os
import json
import yaml
import psycopg2
import pandas as pd
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Set, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

class OntologyBuilder:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.read_conn = psycopg2.connect(**DB_CONFIG) # Separate connection for reading cursors
        self.concepts: Dict[str, Dict] = {} # id -> {label, facets: Set[str]}
        self.edges = defaultdict(int) # (src, dst, type) -> count
        self.paper_evidence = [] # List of tuples for batch insert
        self.min_support = CONFIG['ontology']['min_support']
        
        # Mappings
        self.facet_map = CONFIG['facet_mapping'] # "Research Areas" -> "Area"
        self.relation_map = CONFIG['relation_mapping'] # "Area_System" -> "STUDIES"

    def setup_tables(self):
        """Create necessary tables in the database."""
        with self.conn.cursor() as cur:
            # Edges table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS third_party_aps.ontology_edges (
                    source_id UUID,
                    target_id UUID,
                    relation VARCHAR(50),
                    weight FLOAT,
                    support INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, relation)
                );
            """)
            
            # Evidence table (Paper -> Concept)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS third_party_aps.ontology_paper_evidence (
                    doi VARCHAR(255),
                    concept_id UUID,
                    relation VARCHAR(50), -- MENTIONS (default), USES (for techniques)
                    is_primary BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (doi, concept_id, relation)
                );
            """)
            
            # Core nodes referencing existing tables? 
            # User suggested (:Concept), (:Facet) etc. 
            # We assume physh_concepts is the Concept/Facet source.
            
            self.conn.commit()
            print("Tables initialized.")

    def load_concepts(self):
        """Load concepts and their facets from DB"""
        print("Loading concepts...")
        query = f"SELECT concept_id, concept_label, facet_labels FROM {CONFIG['db']['concepts_table']}"
        
        with self.conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            
        for cid, label, facets_raw in rows:
            # facets_raw might be a list or json string depending on how it was stored
            facets = set()
            if isinstance(facets_raw, str):
                try:
                    parsed = json.loads(facets_raw)
                    if isinstance(parsed, list):
                        facets.update(parsed)
                    else:
                        facets.add(parsed)
                except:
                    facets.add(facets_raw)
            elif isinstance(facets_raw, list):
                facets.update(facets_raw)
            
            # Map raw facet labels to internal types (Area, System, etc)
            mapped_facets = set()
            for f in facets:
                if f in self.facet_map:
                    mapped_facets.add(self.facet_map[f])
            
            if mapped_facets:
                self.concepts[str(cid)] = {
                    "label": label,
                    "facets": mapped_facets
                }
        print(f"Loaded {len(self.concepts)} concepts with mapped facets.")

    def process_papers(self):
        """Iterate through papers, extracting concepts and inducing edges."""
        print("Processing papers...")
        
        # Cursor for large dataset
        chunk_size = 1000
        offset = 0
        total_processed = 0
        
        query = f"""
            SELECT doi, classification_schemes 
            FROM {CONFIG['db']['articles_table']} 
            ORDER BY doi
        """
        
        # We'll use a named cursor or just paging. 20k is small enough for paging or one fetch if memory allows.
        # Let's use server-side cursor to be safe.
        with self.read_conn.cursor(name='paper_cursor') as cur:
            cur.execute(query)
            
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                
                for doi, schemes in rows:
                    if not schemes:
                        continue
                        
                    # Parse schemes (it's JSONB or JSON string)
                    if isinstance(schemes, str):
                        try:
                            schemes = json.loads(schemes)
                        except:
                            continue
                    
                    physh = schemes.get("physh", {})
                    concepts_list = physh.get("concepts", [])
                    
                    # Store current paper's concepts by type
                    current_concepts_by_type = defaultdict(list) # Type -> [id]
                    
                    for c in concepts_list:
                        cid = str(c.get("id"))
                        is_primary = c.get("primary", False)
                        
                        if cid not in self.concepts:
                            continue
                            
                        # Add Evidence
                        concept_types = self.concepts[cid]["facets"]
                        
                        # Determine relation: "USES" if Technique, else "MENTIONS"
                        # If a concept has multiple facets, we might add multiple edges or pick one.
                        # Usually Technique is distinct.
                        rel = "MENTIONS"
                        if "Technique" in concept_types:
                            rel = "USES"
                        
                        # We defer insertion to batch
                        # self.paper_evidence.append((doi, cid, rel, is_primary))
                        # actually, let's just insert to array and batch insert later
                        # OR inserting 20k rows is fast.
                        
                        # For co-occurrence, we collect all valid types this concept represents
                        for t in concept_types:
                            current_concepts_by_type[t].append(cid)
                    
                    # Induce Edges
                    # Rules:
                    # Area (A) -> System (S) : STUDIES
                    # System (S) -> Property (P) : EXHIBITS
                    # Technique (T) -> Property (P) : MEASURES
                    # Technique (T) -> System (S) : PROBES
                    
                    # Helper to count
                    def count_pairs(src_type, dst_type, rel_name):
                        srcs = current_concepts_by_type.get(src_type, [])
                        dsts = current_concepts_by_type.get(dst_type, [])
                        for s, d in product(srcs, dsts):
                            if s == d: continue 
                            self.edges[(s, d, rel_name)] += 1

                    count_pairs("Area", "System", "STUDIES")
                    count_pairs("System", "Property", "EXHIBITS")
                    count_pairs("Technique", "Property", "MEASURES")
                    count_pairs("Technique", "System", "PROBES")

                total_processed += len(rows)
                print(f"Processed {total_processed} papers...", end='\r')
        
        print(f"\nFinished processing {total_processed} papers.")       

    def populate_evidence_layer(self):
        """
        Re-scan papers to populate the evidence table.
        Doing this separately or in same pass? 
        Re-doing logic to keep method clean, but for performance usually 1 pass is better.
        Given 20k papers, 1 pass is preferred.
        
        Actually, let's implement the Evidence insertion in `process_papers` BUT buffering 
        might consume memory. 
        Let's utilize a generator or just do a direct INSERT batching.
        """
        # For simplicity in this prompt, I will implement a separate method that does a fresh pass 
        # OR modifies process_papers to yield evidence.
        # Let's modify process_papers to writing evidence to CSV then COPY, or batch insert.
        pass

    def save_edges(self):
        """Filter and save edges to DB and CSV"""
        print("Saving edges...")
        valid_edges = []
        
        # Prepare CSV data
        csv_data = []
        
        for (src, dst, rel), count in self.edges.items():
            if count >= self.min_support:
                # Calculate weight (placeholder logic)
                weight = 0.5 + (0.5 * (count / 100.0)) # normalizing dummy
                weight = min(weight, 1.0)
                
                valid_edges.append((src, dst, rel, weight, count))
                
                src_label = self.concepts[src]['label']
                dst_label = self.concepts[dst]['label']
                csv_data.append({
                    "source_id": src,
                    "source_label": src_label,
                    "target_id": dst,
                    "target_label": dst_label,
                    "relation": rel,
                    "support": count,
                    "weight": weight
                })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        out_csv = Path(__file__).parent / CONFIG['ontology']['output_csv_edges']
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(valid_edges)} edges to {out_csv}")
        
        # Save to DB
        with self.conn.cursor() as cur:
            # Truncate first? User said "Don't delete table", but maybe clean up previous run?
            # "You may create new tables".
            # I will truncate ontology_edges before inserting to avoid duplicates or use ON CONFLICT DO UPDATE.
            cur.execute("TRUNCATE TABLE third_party_aps.ontology_edges;")
            
            args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in valid_edges)
            if valid_edges:
                cur.execute("INSERT INTO third_party_aps.ontology_edges (source_id, target_id, relation, weight, support) VALUES " + args_str)
            
            self.conn.commit()
            print("Edges committed to DB.")

    def run_evidence_pass(self):
        """
        Separate pass for evidence to handle bulk inserts cleanly.
        """
        print("Generating evidence layer...")
        chunk_size = 1000
        batch_evidence = []
        
        query = f"SELECT doi, classification_schemes FROM {CONFIG['db']['articles_table']}"
        
        with self.read_conn.cursor(name='evidence_cursor') as cur:
            cur.execute(query)
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                
                for doi, schemes in rows:
                    if not schemes or isinstance(schemes, dict): 
                        # schemes from DB might be dict if using jsonb? 
                        # previous method assumed string/json. check.
                        pass
                    
                    if isinstance(schemes, str):
                        try:
                             schemes = json.loads(schemes)
                        except: continue
                    
                    if not isinstance(schemes, dict): continue

                    physh = schemes.get("physh", {})
                    concepts = physh.get("concepts", [])
                    
                    for c in concepts:
                        cid = str(c.get("id"))
                        is_primary = c.get("primary", False)
                        
                        if cid in self.concepts:
                            c_types = self.concepts[cid]["facets"]
                            rel = "USES" if "Technique" in c_types else "MENTIONS"
                            batch_evidence.append((doi, cid, rel, is_primary))
                
                # Flush batch
                if len(batch_evidence) > 5000:
                    self._flush_evidence(batch_evidence)
                    batch_evidence = []
            
            # Flush remaining
            if batch_evidence:
                self._flush_evidence(batch_evidence)
                
    def _flush_evidence(self, batch):
        with self.conn.cursor() as cur:
            args_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x).decode('utf-8') for x in batch)
            cur.execute("INSERT INTO third_party_aps.ontology_paper_evidence (doi, concept_id, relation, is_primary) VALUES " + args_str + " ON CONFLICT DO NOTHING")
            self.conn.commit()      
        print(f"Flushed {len(batch)} evidence records.")

    def close(self):
        self.read_conn.close()
        self.conn.close()

def main():
    builder = OntologyBuilder()
    try:
        builder.setup_tables()
        builder.load_concepts()
        builder.process_papers() # Induces edges
        builder.save_edges()
        
        # User also wants Evidence Layer
        # It's better to clear old evidence first
        with builder.conn.cursor() as cur:
             cur.execute("TRUNCATE TABLE third_party_aps.ontology_paper_evidence;")
             builder.conn.commit()
             
        builder.run_evidence_pass()
        
    finally:
        builder.close()

if __name__ == "__main__":
    main()
