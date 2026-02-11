"""
Extract Concept Co-occurrences from APS Papers

Processes APS articles to extract co-occurrence relationships between
PhySH concepts, building the foundation for the physics ontology.

Usage:
    python extract_cooccurrences.py              # Full extraction
    python extract_cooccurrences.py --incremental  # Only new papers
    python extract_cooccurrences.py --limit 100    # Process N papers (for testing)
    python extract_cooccurrences.py --dry-run      # Show what would be done

This script:
1. Reads papers from third_party_data_source.aps_articles
2. Extracts concept pairs from each paper's PhySH classification
3. Maps concepts to semantic facet types (area, system, property, technique)
4. Counts co-occurrences and stores in database
5. Exports results to CSV for inspection
"""

import os
import json
import uuid
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import psycopg2
from psycopg2.extras import execute_values, Json
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_facet_mapping(conn, config):
    """
    Build a mapping from facet_id -> semantic type (area, system, property, technique).
    
    Returns:
        dict: {facet_id: semantic_type}
    """
    # Get facets from database
    with conn.cursor() as cur:
        cur.execute("""
            SELECT facet_id, facet_label 
            FROM third_party_aps.physh_facets
        """)
        facets = cur.fetchall()
    
    facet_label_to_type = config.get('facet_mappings', {})
    facet_id_to_type = {}
    facet_id_to_label = {}
    
    for facet_id, facet_label in facets:
        facet_id_str = str(facet_id)
        facet_id_to_label[facet_id_str] = facet_label
        # Map to semantic type, default to 'other'
        semantic_type = facet_label_to_type.get(facet_label, 'other')
        facet_id_to_type[facet_id_str] = semantic_type
    
    logger.info(f"Loaded {len(facet_id_to_type)} facet mappings:")
    for facet_id, facet_label in facets:
        semantic_type = facet_id_to_type.get(str(facet_id), 'other')
        logger.info(f"  {facet_label} -> {semantic_type}")
    
    return facet_id_to_type, facet_id_to_label


def get_concept_info(conn):
    """
    Load concept information from database.
    
    Returns:
        dict: {concept_id: {'label': str, 'facet_ids': list}}
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT concept_id, concept_label, facet_ids
            FROM third_party_aps.physh_concepts
        """)
        concepts = cur.fetchall()
    
    concept_info = {}
    for concept_id, label, facet_ids in concepts:
        concept_id_str = str(concept_id)
        # facet_ids is stored as JSONB, parse if string
        if isinstance(facet_ids, str):
            facet_ids = json.loads(facet_ids)
        concept_info[concept_id_str] = {
            'label': label,
            'facet_ids': facet_ids or []
        }
    
    logger.info(f"Loaded {len(concept_info)} concepts from database")
    return concept_info


def get_papers_to_process(conn, config, incremental=False, limit=None):
    """
    Get papers that need to be processed.
    
    Args:
        conn: Database connection
        config: Configuration dict
        incremental: If True, only get papers not yet processed
        limit: Maximum number of papers to return
        
    Returns:
        list: List of (doi, classification_schemes) tuples
    """
    if incremental:
        query = """
            SELECT a.doi, a.classification_schemes
            FROM third_party_data_source.aps_articles a
            LEFT JOIN third_party_aps.physh_processed_papers p ON a.doi = p.paper_doi
            WHERE p.paper_doi IS NULL
            AND a.classification_schemes IS NOT NULL
            ORDER BY a.doi
        """
    else:
        query = """
            SELECT doi, classification_schemes
            FROM third_party_data_source.aps_articles
            WHERE classification_schemes IS NOT NULL
            ORDER BY doi
        """
    
    if limit:
        query += f" LIMIT {limit}"
    
    with conn.cursor() as cur:
        cur.execute(query)
        papers = cur.fetchall()
    
    logger.info(f"Found {len(papers)} papers to process")
    return papers


def extract_concept_pairs_from_paper(paper_doi, classification_schemes, concept_info, facet_id_to_type):
    """
    Extract all concept pairs from a paper's classification.
    
    Args:
        paper_doi: Paper DOI
        classification_schemes: JSON classification data
        concept_info: Dict of concept information
        facet_id_to_type: Mapping from facet ID to semantic type
        
    Returns:
        list: List of dicts with pair information
    """
    pairs = []
    
    # Parse classification if string
    if isinstance(classification_schemes, str):
        try:
            classification_schemes = json.loads(classification_schemes)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse classification for {paper_doi}")
            return pairs
    
    # Get PhySH concepts
    physh_data = classification_schemes.get('physh', {})
    concepts = physh_data.get('concepts', [])
    
    if len(concepts) < 2:
        return pairs  # Need at least 2 concepts for a pair
    
    # Build concept list with metadata
    concept_list = []
    for c in concepts:
        concept_id = c.get('id')
        if not concept_id:
            continue
            
        facet_info = c.get('facet', {})
        facet_id = facet_info.get('id') if isinstance(facet_info, dict) else facet_info
        is_primary = c.get('primary', False)
        
        # Get semantic type
        facet_type = facet_id_to_type.get(str(facet_id), 'other') if facet_id else 'other'
        
        # Get label from concept_info
        info = concept_info.get(str(concept_id), {})
        label = info.get('label', 'Unknown')
        
        concept_list.append({
            'id': str(concept_id),
            'label': label,
            'facet_type': facet_type,
            'is_primary': is_primary
        })
    
    # Generate all pairs
    for c1, c2 in combinations(concept_list, 2):
        # Ensure consistent ordering (source < target by ID)
        if c1['id'] < c2['id']:
            source, target = c1, c2
        else:
            source, target = c2, c1
        
        pairs.append({
            'paper_doi': paper_doi,
            'source_concept_id': source['id'],
            'source_concept_label': source['label'],
            'source_facet_type': source['facet_type'],
            'source_is_primary': source['is_primary'],
            'target_concept_id': target['id'],
            'target_concept_label': target['label'],
            'target_facet_type': target['facet_type'],
            'target_is_primary': target['is_primary']
        })
    
    return pairs


def process_papers(conn, papers, concept_info, facet_id_to_type, config, run_id, dry_run=False):
    """
    Process all papers and extract co-occurrences.
    
    Returns:
        dict: Statistics about the extraction
    """
    # Aggregate co-occurrences
    cooccurrences = defaultdict(lambda: {
        'count': 0,
        'source_label': '',
        'target_label': '',
        'source_facet_type': '',
        'target_facet_type': ''
    })
    
    # Store evidence
    evidence_records = []
    
    batch_size = config.get('cooccurrence', {}).get('batch_size', 1000)
    total_papers = len(papers)
    total_pairs = 0
    
    logger.info(f"Processing {total_papers} papers...")
    
    for i, (doi, classification) in enumerate(papers):
        pairs = extract_concept_pairs_from_paper(
            doi, classification, concept_info, facet_id_to_type
        )
        
        for pair in pairs:
            key = (pair['source_concept_id'], pair['target_concept_id'])
            cooccurrences[key]['count'] += 1
            cooccurrences[key]['source_label'] = pair['source_concept_label']
            cooccurrences[key]['target_label'] = pair['target_concept_label']
            cooccurrences[key]['source_facet_type'] = pair['source_facet_type']
            cooccurrences[key]['target_facet_type'] = pair['target_facet_type']
            
            evidence_records.append({
                'paper_doi': pair['paper_doi'],
                'source_concept_id': pair['source_concept_id'],
                'target_concept_id': pair['target_concept_id'],
                'source_is_primary': pair['source_is_primary'],
                'target_is_primary': pair['target_is_primary']
            })
        
        total_pairs += len(pairs)
        
        # Progress logging
        if (i + 1) % batch_size == 0:
            logger.info(f"Processed {i + 1}/{total_papers} papers ({total_pairs} pairs found)")
    
    logger.info(f"Extraction complete: {len(cooccurrences)} unique pairs from {total_pairs} total co-occurrences")
    
    if dry_run:
        logger.info("DRY RUN - not saving to database")
        return {
            'papers_processed': total_papers,
            'unique_pairs': len(cooccurrences),
            'total_cooccurrences': total_pairs
        }
    
    # Save to database
    min_support = config.get('cooccurrence', {}).get('min_support', 5)
    
    # Filter by minimum support
    filtered_cooccurrences = {
        k: v for k, v in cooccurrences.items() 
        if v['count'] >= min_support
    }
    
    logger.info(f"After min_support={min_support} filter: {len(filtered_cooccurrences)} pairs")
    
    # Insert co-occurrences
    save_cooccurrences(conn, filtered_cooccurrences)
    
    # Insert evidence (only for pairs that meet threshold)
    valid_pairs = set(filtered_cooccurrences.keys())
    filtered_evidence = [
        e for e in evidence_records 
        if (e['source_concept_id'], e['target_concept_id']) in valid_pairs
    ]
    save_evidence(conn, filtered_evidence)
    
    # Mark papers as processed
    save_processed_papers(conn, [p[0] for p in papers], run_id)
    
    return {
        'papers_processed': total_papers,
        'unique_pairs': len(filtered_cooccurrences),
        'total_cooccurrences': total_pairs,
        'evidence_records': len(filtered_evidence)
    }


def save_cooccurrences(conn, cooccurrences):
    """Save co-occurrence counts to database."""
    if not cooccurrences:
        return
    
    values = []
    for (source_id, target_id), data in cooccurrences.items():
        values.append((
            source_id,
            data['source_label'],
            data['source_facet_type'],
            target_id,
            data['target_label'],
            data['target_facet_type'],
            data['count']
        ))
    
    insert_sql = """
        INSERT INTO third_party_aps.physh_concept_cooccurrences
            (source_concept_id, source_concept_label, source_facet_type,
             target_concept_id, target_concept_label, target_facet_type,
             cooccurrence_count)
        VALUES %s
        ON CONFLICT (source_concept_id, target_concept_id) DO UPDATE SET
            cooccurrence_count = third_party_aps.physh_concept_cooccurrences.cooccurrence_count + EXCLUDED.cooccurrence_count,
            updated_at = CURRENT_TIMESTAMP
    """
    
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values)
    conn.commit()
    
    logger.info(f"Saved {len(values)} co-occurrence records")


def save_evidence(conn, evidence_records):
    """Save paper-concept evidence to database."""
    if not evidence_records:
        return
    
    values = [
        (e['paper_doi'], e['source_concept_id'], e['target_concept_id'],
         e['source_is_primary'], e['target_is_primary'])
        for e in evidence_records
    ]
    
    insert_sql = """
        INSERT INTO third_party_aps.physh_paper_concept_evidence
            (paper_doi, source_concept_id, target_concept_id,
             source_is_primary, target_is_primary)
        VALUES %s
        ON CONFLICT (paper_doi, source_concept_id, target_concept_id) DO NOTHING
    """
    
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values)
    conn.commit()
    
    logger.info(f"Saved {len(values)} evidence records")


def save_processed_papers(conn, dois, run_id):
    """Mark papers as processed."""
    values = [(doi, str(run_id)) for doi in dois]
    
    insert_sql = """
        INSERT INTO third_party_aps.physh_processed_papers
            (paper_doi, extraction_run_id)
        VALUES %s
        ON CONFLICT (paper_doi) DO UPDATE SET
            extraction_run_id = EXCLUDED.extraction_run_id,
            processed_at = CURRENT_TIMESTAMP
    """
    
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, values)
    conn.commit()


def create_extraction_run(conn, config, run_type):
    """Create a new extraction run record."""
    run_id = uuid.uuid4()
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO third_party_aps.physh_extraction_metadata
                (run_id, run_type, config_snapshot, status)
            VALUES (%s, %s, %s, 'running')
        """, (str(run_id), run_type, Json(config)))
    conn.commit()
    
    return run_id


def complete_extraction_run(conn, run_id, stats):
    """Mark extraction run as complete."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE third_party_aps.physh_extraction_metadata
            SET 
                papers_processed = %s,
                cooccurrences_found = %s,
                completed_at = CURRENT_TIMESTAMP,
                status = 'completed'
            WHERE run_id = %s
        """, (stats['papers_processed'], stats['unique_pairs'], str(run_id)))
    conn.commit()


def export_to_csv(conn, config):
    """Export co-occurrences to CSV for inspection."""
    output_dir = Path(__file__).parent / config.get('export', {}).get('output_dir', 'output')
    output_dir.mkdir(exist_ok=True)
    
    csv_files = config.get('export', {}).get('csv_files', {})
    
    # Export co-occurrences
    cooccur_file = output_dir / csv_files.get('cooccurrences', 'concept_cooccurrences.csv')
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                source_concept_id, source_concept_label, source_facet_type,
                target_concept_id, target_concept_label, target_facet_type,
                cooccurrence_count, weight
            FROM third_party_aps.physh_concept_cooccurrences
            ORDER BY cooccurrence_count DESC
        """)
        rows = cur.fetchall()
    
    import csv
    with open(cooccur_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'source_concept_id', 'source_concept_label', 'source_facet_type',
            'target_concept_id', 'target_concept_label', 'target_facet_type',
            'cooccurrence_count', 'weight'
        ])
        writer.writerows(rows)
    
    logger.info(f"Exported {len(rows)} co-occurrences to {cooccur_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract concept co-occurrences from APS papers')
    parser.add_argument('--incremental', action='store_true', 
                        help='Only process papers not yet processed')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of papers to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Do not save to database, just show statistics')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export existing data to CSV, no extraction')
    args = parser.parse_args()
    
    config = load_config()
    
    logger.info("Connecting to database...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"✓ Connected to {DB_CONFIG['host']}")
    except psycopg2.Error as e:
        logger.error(f"✗ Failed to connect: {e}")
        return
    
    try:
        if args.export_only:
            export_to_csv(conn, config)
            return
        
        # Load mappings
        facet_id_to_type, facet_id_to_label = get_facet_mapping(conn, config)
        concept_info = get_concept_info(conn)
        
        # Get papers to process
        papers = get_papers_to_process(
            conn, config, 
            incremental=args.incremental, 
            limit=args.limit
        )
        
        if not papers:
            logger.info("No papers to process")
            return
        
        # Create extraction run
        run_type = 'incremental' if args.incremental else 'full'
        run_id = create_extraction_run(conn, config, run_type) if not args.dry_run else None
        
        # Process papers
        stats = process_papers(
            conn, papers, concept_info, facet_id_to_type, 
            config, run_id, dry_run=args.dry_run
        )
        
        if not args.dry_run:
            complete_extraction_run(conn, run_id, stats)
            export_to_csv(conn, config)
        
        logger.info(f"\n=== Extraction Summary ===")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
