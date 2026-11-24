"""
Download Random Triplet Sample
===============================
Pulls N random rows from graph_rag.graph_triplets_2025_09_09 into a CSV file
for subsequent evaluation by evaluate_triplets.py.

Usage:
    python3 download_sample.py                    # uses config.yaml defaults
    python3 download_sample.py --n_rows 200       # override sample size
    python3 download_sample.py --output my.csv    # override output path
"""

import argparse
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ── Load environment ──────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_engine():
    user = quote_plus(os.getenv("DB_USER"))
    password = quote_plus(os.getenv("DB_PASSWORD"))
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url)


def main():
    cfg = load_config()
    sample_cfg = cfg["sample"]

    parser = argparse.ArgumentParser(description="Download random triplet sample")
    parser.add_argument("--n_rows", type=int, default=sample_cfg["n_rows"],
                        help="Number of random rows to sample")
    parser.add_argument("--output", type=str, default=sample_cfg["output_file"],
                        help="Output CSV file path")
    parser.add_argument("--table", type=str, default=sample_cfg["table"],
                        help="Source database table")
    args = parser.parse_args()

    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.n_rows} random rows from {args.table} ...")
    engine = get_engine()

    query = f"""
        SELECT *
        FROM {args.table}
        ORDER BY RANDOM()
        LIMIT {args.n_rows}
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample (first 3 rows):")
    print(df[["subject", "predicate", "object", "confidence", "paper_id"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
