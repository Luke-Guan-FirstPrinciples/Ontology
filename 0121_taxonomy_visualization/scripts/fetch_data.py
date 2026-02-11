#!/usr/bin/env python3
"""
Fetch PhySH Taxonomy Data for Visualization

This script fetches concept data from the database and exports it as JSON files
optimized for the visualization frontend.

Output files (in ../data/):
- concepts.json: Core taxonomy data
- stats.json: Frequency statistics
- papers.json: Paper metadata and label mappings
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory for shared imports
# 0121_taxonomy_visualization/scripts/ -> 0121_taxonomy_visualization/ -> taxonomy/ -> 0119_taxonomy_similarity
shared_path = Path(__file__).resolve().parent.parent.parent / "0119_taxonomy_similarity"
sys.path.insert(0, str(shared_path))

try:
    from shared.db_utils import get_db_connection, load_config, get_schema_names
except ImportError:
    # Fallback if the path resolution fails
    sys.path.append("/Users/firstprinciplesextralaptop2/code/2026/taxonomy/0119_taxonomy_similarity")
    from shared.db_utils import get_db_connection, load_config, get_schema_names

def fetch_concepts(conn, schema: str, limit: int = None) -> list:
    """Fetch concepts from the database."""
    query = f"""
        SELECT 
            concept_id,
            label,
            definition,
            depth,
            parent_id,
            facet_labels,
            facet_ids,
            discipline_labels,
            discipline_ids,
            concept_paths,
            related_concept_ids,
            narrower_concept_ids,
            broader_concept_ids,
            exclude_from_indexing
        FROM {schema}.physh_concepts
        ORDER BY depth ASC, label ASC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    
    concepts = []
    for idx, row in enumerate(rows):
        concept = dict(zip(columns, row))
        concepts.append(concept)
    
    print(f"Fetched {len(concepts)} concepts from {schema}.physh_concepts")
    return concepts

def fetch_stats(conn, schema: str) -> list:
    """Fetch statistics from the enriched concepts table."""
    query = f"""
        SELECT 
            concept_id,
            aps_frequency
        FROM {schema}.physh_concepts_enriched
        WHERE aps_frequency > 0
        ORDER BY concept_id ASC
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        
        stats = [dict(zip(columns, row)) for row in rows]
        print(f"Fetched {len(stats)} stats from {schema}.physh_concepts_enriched")
        return stats
    except Exception as e:
        print(f"Warning: Could not fetch stats: {e}")
        return []

def normalize_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [item for item in parsed if item]
        except json.JSONDecodeError:
            pass
    return [value] if value else []

def fetch_papers(conn, labels_schema: str, raw_schema: str, limit: int = None) -> list:
    """Fetch paper labels (DOI -> concept/discipline IDs) with metadata."""
    query = f"""
        SELECT
            l.doi,
            l.concept_ids,
            l.discipline_ids,
            a.title_value,
            a.date
        FROM {labels_schema}.aps_paper_labels l
        LEFT JOIN {raw_schema}.aps_articles a ON a.doi = l.doi
        ORDER BY l.doi ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    papers = []
    for doi, concept_ids, discipline_ids, title, date_value in rows:
        year = None
        if isinstance(date_value, datetime):
            year = date_value.year
        elif date_value:
            try:
                year = int(str(date_value)[:4])
            except ValueError:
                year = None

        papers.append({
            "doi": doi,
            "title": title,
            "year": year,
            "c": normalize_json_list(concept_ids), # Shortened key
            "d": normalize_json_list(discipline_ids), # Shortened key
        })

    print(f"Fetched {len(papers)} papers from {labels_schema}.aps_paper_labels")
    return papers

def save_js(data, filename: str, var_name: str):
    """Save data to JS file with global variable assignment."""
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / filename
    
    # Efficient writing: "window.VAR = " + json_dump
    with open(output_path, 'w') as f:
        f.write(f"window.{var_name} = ")
        json.dump(data, f, separators=(',', ':'))
        f.write(";")
    
    size_kb = output_path.stat().st_size / 1024
    print(f"Saved {filename} ({size_kb:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description="Fetch PhySH data")
    parser.add_argument("--limit", type=int, help="Limit concepts")
    parser.add_argument("--paper-limit", type=int, help="Limit papers")
    args = parser.parse_args()
    
    config = load_config()
    schemas = get_schema_names(config)
    
    physh_raw = schemas.get("physh_raw", "third_party_aps_physh_raw")
    physh_transformed = schemas.get("physh_transformed", "third_party_aps_physh_transformed")
    harvest_raw = schemas.get("harvest_raw", "third_party_aps_harvest_raw")
    harvest_transformed = schemas.get("harvest_transformed", "third_party_aps_harvest_transformed")
    
    conn = get_db_connection()
    
    print("Fetching data...")
    concepts = fetch_concepts(conn, physh_raw, args.limit)
    stats = fetch_stats(conn, physh_transformed)
    papers = fetch_papers(conn, harvest_transformed, harvest_raw, args.paper_limit)
    
    # Post-process concepts to minimize JSON size
    minified_concepts = []
    for c in concepts:
        minified_concepts.append({
            "id": c["concept_id"],
            "l": c["label"],
            "d": c["depth"],
            "p": c["parent_id"],
            "fl": normalize_json_list(c["facet_labels"]),
            "dl": normalize_json_list(c["discipline_labels"]),
            "def": c["definition"]
        })
        
    save_js(minified_concepts, "concepts.js", "CONCEPTS")
    save_js(stats, "stats.js", "STATS")
    save_js(papers, "papers.js", "PAPERS")
    
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
