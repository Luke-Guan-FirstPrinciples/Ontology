"""
Create Ontology Tables in PostgreSQL

Creates the database schema for storing the induced physics ontology,
including co-occurrence counts, semantic edges, and evidence links.

Usage:
    python create_ontology_tables.py

Tables created:
    - physh_concept_cooccurrences: Raw co-occurrence counts between concepts
    - physh_ontology_edges: Semantic edges (STUDIES, EXHIBITS, MEASURES, PROBES)
    - physh_paper_concept_evidence: Links papers to concept pairs as evidence
    - physh_extraction_metadata: Tracks extraction runs for incremental updates
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}


def create_tables(conn):
    """Create all ontology-related tables."""
    
    create_sql = """
    -- ============================================================
    -- PhySH Ontology Tables
    -- ============================================================
    
    -- Raw concept co-occurrence counts
    -- Stores how often two concepts appear together in papers
    CREATE TABLE IF NOT EXISTS third_party_aps.physh_concept_cooccurrences (
        id SERIAL PRIMARY KEY,
        
        -- Source concept (lower alphabetically by ID to avoid duplicates)
        source_concept_id UUID NOT NULL,
        source_concept_label TEXT NOT NULL,
        source_facet_type TEXT NOT NULL,  -- area, system, property, technique, other
        
        -- Target concept
        target_concept_id UUID NOT NULL,
        target_concept_label TEXT NOT NULL,
        target_facet_type TEXT NOT NULL,
        
        -- Co-occurrence statistics
        cooccurrence_count INTEGER NOT NULL DEFAULT 0,
        
        -- Normalized weight (0-1, calculated after all counts are in)
        weight FLOAT,
        
        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Ensure unique pairs (source always < target alphabetically by ID)
        UNIQUE(source_concept_id, target_concept_id)
    );
    
    -- Semantic ontology edges
    -- Derived from co-occurrences based on facet types
    CREATE TABLE IF NOT EXISTS third_party_aps.physh_ontology_edges (
        id SERIAL PRIMARY KEY,
        
        -- Source concept
        source_concept_id UUID NOT NULL,
        source_concept_label TEXT NOT NULL,
        
        -- Relation type: STUDIES, EXHIBITS, MEASURES, PROBES, RELATED_TO
        relation_type TEXT NOT NULL,
        
        -- Target concept
        target_concept_id UUID NOT NULL,
        target_concept_label TEXT NOT NULL,
        
        -- Edge strength
        support INTEGER NOT NULL,  -- Number of papers supporting this edge
        weight FLOAT,              -- Normalized weight
        
        -- Confidence metrics (for future use)
        confidence FLOAT,
        
        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Unique edge per relation type
        UNIQUE(source_concept_id, relation_type, target_concept_id)
    );
    
    -- Evidence table linking papers to concept pairs
    -- Answers "Why do we believe this edge exists?"
    CREATE TABLE IF NOT EXISTS third_party_aps.physh_paper_concept_evidence (
        id SERIAL PRIMARY KEY,
        
        -- The paper providing evidence
        paper_doi TEXT NOT NULL,
        
        -- The concept pair
        source_concept_id UUID NOT NULL,
        target_concept_id UUID NOT NULL,
        
        -- Whether source concept was primary in this paper
        source_is_primary BOOLEAN DEFAULT FALSE,
        target_is_primary BOOLEAN DEFAULT FALSE,
        
        -- Timestamp
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Unique evidence per paper-pair
        UNIQUE(paper_doi, source_concept_id, target_concept_id)
    );
    
    -- Extraction metadata for incremental updates
    CREATE TABLE IF NOT EXISTS third_party_aps.physh_extraction_metadata (
        id SERIAL PRIMARY KEY,
        
        -- Extraction run info
        run_id UUID NOT NULL UNIQUE,
        run_type TEXT NOT NULL,  -- 'full' or 'incremental'
        
        -- Statistics
        papers_processed INTEGER DEFAULT 0,
        cooccurrences_found INTEGER DEFAULT 0,
        edges_created INTEGER DEFAULT 0,
        
        -- Configuration used
        config_snapshot JSONB,
        
        -- Timing
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        
        -- Status
        status TEXT DEFAULT 'running'  -- running, completed, failed
    );
    
    -- Track which papers have been processed (for incremental mode)
    CREATE TABLE IF NOT EXISTS third_party_aps.physh_processed_papers (
        paper_doi TEXT PRIMARY KEY,
        extraction_run_id UUID NOT NULL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- ============================================================
    -- Indexes for performance
    -- ============================================================
    
    -- Co-occurrence lookups
    CREATE INDEX IF NOT EXISTS idx_cooccur_source 
        ON third_party_aps.physh_concept_cooccurrences(source_concept_id);
    CREATE INDEX IF NOT EXISTS idx_cooccur_target 
        ON third_party_aps.physh_concept_cooccurrences(target_concept_id);
    CREATE INDEX IF NOT EXISTS idx_cooccur_facets 
        ON third_party_aps.physh_concept_cooccurrences(source_facet_type, target_facet_type);
    CREATE INDEX IF NOT EXISTS idx_cooccur_count 
        ON third_party_aps.physh_concept_cooccurrences(cooccurrence_count DESC);
    
    -- Ontology edge lookups
    CREATE INDEX IF NOT EXISTS idx_edge_source 
        ON third_party_aps.physh_ontology_edges(source_concept_id);
    CREATE INDEX IF NOT EXISTS idx_edge_target 
        ON third_party_aps.physh_ontology_edges(target_concept_id);
    CREATE INDEX IF NOT EXISTS idx_edge_relation 
        ON third_party_aps.physh_ontology_edges(relation_type);
    CREATE INDEX IF NOT EXISTS idx_edge_support 
        ON third_party_aps.physh_ontology_edges(support DESC);
    
    -- Evidence lookups
    CREATE INDEX IF NOT EXISTS idx_evidence_paper 
        ON third_party_aps.physh_paper_concept_evidence(paper_doi);
    CREATE INDEX IF NOT EXISTS idx_evidence_concepts 
        ON third_party_aps.physh_paper_concept_evidence(source_concept_id, target_concept_id);
    """
    
    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()
    print("✓ Ontology tables created successfully")


def print_table_info(conn):
    """Print information about the created tables."""
    tables = [
        "physh_concept_cooccurrences",
        "physh_ontology_edges", 
        "physh_paper_concept_evidence",
        "physh_extraction_metadata",
        "physh_processed_papers"
    ]
    
    print("\n=== Ontology Tables ===")
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"""
                SELECT COUNT(*) FROM third_party_aps.{table}
            """)
            count = cur.fetchone()[0]
            print(f"  {table}: {count} rows")


def main():
    """Main execution."""
    print("Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"✓ Connected to {DB_CONFIG['host']}")
    except psycopg2.Error as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    try:
        create_tables(conn)
        print_table_info(conn)
        print("\n✓ Setup complete!")
    except Exception as e:
        print(f"✗ Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
