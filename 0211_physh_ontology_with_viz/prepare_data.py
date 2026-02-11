"""
Prepare ontology data for the v3 web visualization.
Reads ontology_edges.csv and exports JS data files.
Optionally enriches with DB metadata if available.
"""

import csv
import json
import os
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# Facet inference from relation type (source -> target)
RELATION_FACET_MAP = {
    "STUDIES":  ("Research Areas", "Physical Systems"),
    "EXHIBITS": ("Physical Systems", "Properties"),
    "MEASURES": ("Techniques", "Properties"),
    "PROBES":   ("Techniques", "Physical Systems"),
}

SCRIPT_DIR = Path(__file__).parent


def load_csv(csv_path, min_support=0):
    """Load edges from CSV file."""
    edges = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            support = int(row["support"])
            if support >= min_support:
                edges.append({
                    "s": row["source_id"],
                    "sl": row["source_label"],
                    "t": row["target_id"],
                    "tl": row["target_label"],
                    "r": row["relation"],
                    "sup": support,
                    "w": round(float(row["weight"]), 3),
                })
    return edges


def try_enrich_from_db(nodes):
    """Try to enrich node data with disciplines from DB. Silently skip if unavailable."""
    try:
        import psycopg2
        from dotenv import load_dotenv

        env_path = SCRIPT_DIR.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "postgres"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
        }

        conn = psycopg2.connect(**db_config)
        node_ids = list(nodes.keys())

        # Fetch in batches
        batch_size = 500
        for i in range(0, len(node_ids), batch_size):
            batch = node_ids[i:i + batch_size]
            placeholders = ",".join(["%s"] * len(batch))
            query = f"""
                SELECT concept_id, discipline_labels, facet_labels
                FROM third_party_aps.physh_concepts
                WHERE concept_id IN ({placeholders})
            """
            with conn.cursor() as cur:
                cur.execute(query, batch)
                for row in cur.fetchall():
                    cid, disc_labels, facet_labels = row
                    if cid in nodes:
                        if disc_labels:
                            dl = disc_labels if isinstance(disc_labels, list) else json.loads(disc_labels) if isinstance(disc_labels, str) else []
                            nodes[cid]["dl"] = dl
                        if facet_labels:
                            fl = facet_labels if isinstance(facet_labels, list) else json.loads(facet_labels) if isinstance(facet_labels, str) else []
                            nodes[cid]["fl"] = fl

        conn.close()
        print("  Enriched node data from database.")
    except Exception as e:
        print(f"  DB enrichment skipped: {e}")


def build_data(edges):
    """Build nodes and edges data structures."""
    nodes = {}
    relation_counts = Counter()
    node_degree = Counter()

    for edge in edges:
        sid, tid, rel = edge["s"], edge["t"], edge["r"]

        # Infer facets from relation type
        src_facet, tgt_facet = RELATION_FACET_MAP.get(rel, ("Unknown", "Unknown"))

        if sid not in nodes:
            nodes[sid] = {
                "id": sid,
                "l": edge["sl"],
                "fl": [src_facet],
                "dl": [],
            }
        if tid not in nodes:
            nodes[tid] = {
                "id": tid,
                "l": edge["tl"],
                "fl": [tgt_facet],
                "dl": [],
            }

        relation_counts[rel] += 1
        node_degree[sid] += 1
        node_degree[tid] += 1

    # Add degree to nodes
    for nid, node in nodes.items():
        node["deg"] = node_degree.get(nid, 0)

    return nodes, relation_counts


def export_js(data, var_name, output_path):
    """Export data as a JS file with window.VAR = ...;"""
    json_str = json.dumps(data, separators=(",", ":"))
    with open(output_path, "w") as f:
        f.write(f"window.{var_name}={json_str};\n")
    print(f"  Exported {output_path.name} ({len(json_str) // 1024} KB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare ontology data for v3 visualization")
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "ontology_edges.csv"),
                        help="Path to ontology_edges.csv")
    parser.add_argument("--min-support", type=int, default=0,
                        help="Minimum support threshold for including edges (default: 0, include all)")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "v3" / "data"),
                        help="Output directory for JS data files")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip database enrichment")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading edges from {args.csv} (min support: {args.min_support})...")
    edges = load_csv(args.csv, min_support=args.min_support)
    print(f"  Loaded {len(edges)} edges")

    print("Building node data...")
    nodes, relation_counts = build_data(edges)
    print(f"  Built {len(nodes)} unique nodes")

    # Try DB enrichment
    if not args.no_db:
        print("Attempting DB enrichment...")
        try_enrich_from_db(nodes)

    # Export
    print("Exporting data files...")
    nodes_list = list(nodes.values())
    export_js(nodes_list, "NODES", output_dir / "nodes.js")
    export_js(edges, "EDGES", output_dir / "edges.js")

    # Export metadata/stats
    stats = {
        "totalNodes": len(nodes),
        "totalEdges": len(edges),
        "relations": dict(relation_counts),
    }
    export_js(stats, "GRAPH_STATS", output_dir / "stats.js")

    print(f"\nDone! Data exported to {output_dir}/")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Relations: {dict(relation_counts)}")


if __name__ == "__main__":
    main()
