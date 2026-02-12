#!/usr/bin/env python3
"""
Wikidata Physics Knowledge Graph Explorer
==========================================

Exploratory script to understand what relationships Wikidata actually
encodes for physics concepts.

Approach:
  1. Start with ~60 curated seed entities (particles, theories, equations, etc.)
  2. Query Wikidata for ALL outgoing triples from these entities
  3. Analyze which predicates appear most often
  4. Focus on inter-entity links (entity→entity, not entity→literal)
  5. Separate hierarchical predicates (instance of, subclass of) from
     semantically rich ones (has part, uses, has cause, etc.)

Output:
  - Console: readable summary of findings
  - JSON: full triple data for further analysis
  - TSV: human-readable triple list

Usage:
  python explore_physics_kg.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

from seed_entities import SEED_ENTITIES, PHYSICS_PREDICATES_OF_INTEREST
from wikidata_sparql import (
    query_all_outgoing_triples,
    query_predicate_frequency,
    query_inter_entity_links,
    query_specific_predicates,
    query_physics_neighbors,
)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BATCH_SIZE = 15  # How many entities per SPARQL query (to stay under URL limits)

HIERARCHICAL_PREDS = {"P31", "P279"}  # instance of, subclass of


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def batched(items, n):
    """Yield successive n-sized chunks from items."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def shorten_value(val: str) -> str:
    """Make long URIs or values readable."""
    if val and val.startswith("http://www.wikidata.org/entity/"):
        return val.rsplit("/", 1)[-1]
    if val and len(val) > 80:
        return val[:77] + "..."
    return val or ""


# ─────────────────────────────────────────────────────────────────────
# Step 1: Predicate Frequency Analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_predicate_frequency(all_qids: list[str]) -> list[dict]:
    """Which predicates appear most often across our physics entities?"""
    print("\n" + "=" * 70)
    print("STEP 1: Predicate Frequency Analysis")
    print("=" * 70)
    print(f"Querying predicate frequencies across {len(all_qids)} entities...")

    all_freq = Counter()
    prop_labels = {}

    for batch in batched(all_qids, BATCH_SIZE):
        freq = query_predicate_frequency(batch, limit=200)
        for f in freq:
            all_freq[f["prop"]] += f["count"]
            prop_labels[f["prop"]] = f["propLabel"]

    sorted_freq = sorted(all_freq.items(), key=lambda x: -x[1])

    print(f"\nFound {len(sorted_freq)} distinct predicates.\n")
    print(f"{'Rank':<5} {'PID':<8} {'Predicate':<40} {'Count':<6} {'Type'}")
    print("-" * 75)

    freq_data = []
    for i, (pid, count) in enumerate(sorted_freq[:50], 1):
        label = prop_labels.get(pid, "?")
        kind = "HIERARCHY" if pid in HIERARCHICAL_PREDS else (
            "KNOWN" if pid in PHYSICS_PREDICATES_OF_INTEREST else "other"
        )
        print(f"{i:<5} {pid:<8} {label:<40} {count:<6} {kind}")
        freq_data.append({
            "rank": i, "pid": pid, "label": label,
            "count": count, "type": kind
        })

    # Summary stats
    hier_count = sum(c for p, c in sorted_freq if p in HIERARCHICAL_PREDS)
    total_count = sum(c for _, c in sorted_freq)
    print(f"\n  Total triples:       {total_count}")
    print(f"  Hierarchical:        {hier_count} ({100*hier_count/max(total_count,1):.1f}%)")
    print(f"  Non-hierarchical:    {total_count - hier_count} ({100*(total_count-hier_count)/max(total_count,1):.1f}%)")

    return freq_data


# ─────────────────────────────────────────────────────────────────────
# Step 2: Inter-Entity Links (the good stuff)
# ─────────────────────────────────────────────────────────────────────

def analyze_inter_entity_links(all_qids: list[str]) -> list[dict]:
    """Find links where both subject and object are in our physics seed set."""
    print("\n" + "=" * 70)
    print("STEP 2: Inter-Entity Links (physics ↔ physics)")
    print("=" * 70)
    print("Looking for relationships between our seed physics entities...\n")

    links = query_inter_entity_links(all_qids, limit=3000)

    if not links:
        print("  No inter-entity links found. (Unusual—check QIDs.)")
        return []

    # Separate hierarchical from rich predicates
    hier_links = [l for l in links if l["prop"] in HIERARCHICAL_PREDS]
    rich_links = [l for l in links if l["prop"] not in HIERARCHICAL_PREDS]

    print(f"  Total inter-entity links found: {len(links)}")
    print(f"  Hierarchical (instance of / subclass of): {len(hier_links)}")
    print(f"  Semantically rich: {len(rich_links)}")

    # Print hierarchical links
    print(f"\n── Hierarchical Links ({'showing all' if len(hier_links) <= 30 else 'top 30'}) ──")
    for l in hier_links[:30]:
        print(f"  {l['entityLabel']:<30} ──[{l['propLabel']}]──▶  {l['targetLabel']}")

    # Print rich links (these are what we really want)
    print(f"\n── Semantically Rich Links ({'showing all' if len(rich_links) <= 50 else 'top 50'}) ──")
    for l in rich_links[:50]:
        print(f"  {l['entityLabel']:<30} ──[{l['propLabel']}]──▶  {l['targetLabel']}")

    # Predicate breakdown for rich links
    rich_pred_counts = Counter(l["propLabel"] for l in rich_links)
    if rich_pred_counts:
        print("\n── Predicate breakdown (non-hierarchical inter-entity) ──")
        for pred, count in rich_pred_counts.most_common(20):
            print(f"  {pred:<40} {count}")

    return links


# ─────────────────────────────────────────────────────────────────────
# Step 3: Sample Triples (all outgoing, for a subset of entities)
# ─────────────────────────────────────────────────────────────────────

def sample_all_triples(sample_qids: list[str], sample_labels: dict) -> list[dict]:
    """Get ALL outgoing triples for a small sample of entities."""
    print("\n" + "=" * 70)
    print("STEP 3: Full Triple Dump (sample entities)")
    print("=" * 70)

    sample_names = [sample_labels.get(q, q) for q in sample_qids]
    print(f"Sampling all triples for: {', '.join(sample_names)}\n")

    all_triples = []
    for batch in batched(sample_qids, BATCH_SIZE):
        triples = query_all_outgoing_triples(batch, limit=3000)
        all_triples.extend(triples)

    # Group by entity
    by_entity = defaultdict(list)
    for t in all_triples:
        by_entity[t["entityLabel"]].append(t)

    for entity_label, triples in sorted(by_entity.items()):
        print(f"\n┌─ {entity_label} ({len(triples)} triples) ─────────────────")
        for t in triples[:25]:
            val_display = t["valueLabel"] or shorten_value(t["value"])
            kind = "⊡" if t["prop"] in HIERARCHICAL_PREDS else "◆"
            print(f"│ {kind} ──[{t['propLabel']}]──▶  {val_display}")
        if len(triples) > 25:
            print(f"│ ... and {len(triples) - 25} more triples")
        print("└" + "─" * 50)

    return all_triples


# ─────────────────────────────────────────────────────────────────────
# Step 4: Physics Neighbors (1-hop expansion)
# ─────────────────────────────────────────────────────────────────────

def discover_physics_neighbors(sample_qids: list[str]) -> list[dict]:
    """Find physics-related entities connected to our seed set."""
    print("\n" + "=" * 70)
    print("STEP 4: Physics Neighbor Discovery (1-hop)")
    print("=" * 70)
    print("Finding physics entities connected to our seed set...\n")

    neighbors = query_physics_neighbors(sample_qids, limit=300)

    if not neighbors:
        print("  No physics neighbors found in 1-hop expansion.")
        return []

    # Deduplicate and count
    neighbor_counts = Counter(
        (n["neighbor"], n["neighborLabel"]) for n in neighbors
    )
    connecting_preds = Counter(n["connectingPropLabel"] for n in neighbors)

    print(f"  Discovered {len(neighbor_counts)} unique physics neighbors.\n")

    print("── Most connected physics neighbors ──")
    for (qid, label), count in neighbor_counts.most_common(30):
        print(f"  {label:<40} ({qid})  connections: {count}")

    print("\n── Connecting predicates ──")
    for pred, count in connecting_preds.most_common(15):
        print(f"  {pred:<40} {count}")

    return neighbors


# ─────────────────────────────────────────────────────────────────────
# Step 5: Focused Predicate Exploration
# ─────────────────────────────────────────────────────────────────────

def explore_interesting_predicates(all_qids: list[str]) -> list[dict]:
    """Query specifically for the predicates we think are ontologically interesting."""
    print("\n" + "=" * 70)
    print("STEP 5: Focused Predicate Exploration")
    print("=" * 70)

    interesting_pids = [
        "P361",   # part of
        "P527",   # has part(s)
        "P1552",  # has characteristic
        "P2283",  # uses
        "P1535",  # used by
        "P828",   # has cause
        "P1542",  # has effect
        "P737",   # influenced by
        "P144",   # based on
        "P1269",  # facet of
        "P2579",  # studied in
        "P61",    # discoverer or inventor
    ]

    pid_labels = {pid: PHYSICS_PREDICATES_OF_INTEREST.get(pid, pid) for pid in interesting_pids}
    print(f"Querying {len(interesting_pids)} semantically rich predicates...\n")

    all_results = []
    for batch in batched(all_qids, BATCH_SIZE):
        results = query_specific_predicates(batch, interesting_pids, limit=2000)
        all_results.extend(results)

    if not all_results:
        print("  No results for these predicates. They may be sparsely used.")
        return []

    # Group by predicate
    by_pred = defaultdict(list)
    for r in all_results:
        by_pred[r["propLabel"]].append(r)

    for pred_label, triples in sorted(by_pred.items(), key=lambda x: -len(x[1])):
        print(f"\n── {pred_label} ({len(triples)} triples) ──")
        for t in triples[:15]:
            val = t["valueLabel"] or shorten_value(t["value"])
            print(f"  {t['entityLabel']:<30} ──▶  {val}")
        if len(triples) > 15:
            print(f"  ... and {len(triples) - 15} more")

    return all_results


# ─────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────

def save_json(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filepath}")


def save_tsv(triples, filename, columns=None):
    """Save triples as a readable TSV file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not triples:
        return
    if columns is None:
        columns = list(triples[0].keys())
    with open(filepath, "w") as f:
        f.write("\t".join(columns) + "\n")
        for t in triples:
            row = "\t".join(str(t.get(c, "")) for c in columns)
            f.write(row + "\n")
    print(f"  Saved: {filepath}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Wikidata Physics Knowledge Graph Explorer                 ║")
    print("║   Exploring what relationships Wikidata encodes for physics ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Seed entities: {len(SEED_ENTITIES)}")

    ensure_output_dir()

    # Build QID list and reverse lookup
    all_qids = list(SEED_ENTITIES.values())
    qid_to_label = {v: k for k, v in SEED_ENTITIES.items()}

    # Pick a representative sample for the full triple dump
    sample_keys = [
        "electron", "photon", "Standard Model", "Schrödinger equation",
        "speed of light", "quantum entanglement", "black hole",
        "electromagnetism", "Planck constant", "Higgs boson",
    ]
    sample_qids = [SEED_ENTITIES[k] for k in sample_keys if k in SEED_ENTITIES]

    # ── Run analyses ──────────────────────────────────────────────────

    # 1. Predicate frequency
    freq_data = analyze_predicate_frequency(all_qids)

    # 2. Inter-entity links
    inter_links = analyze_inter_entity_links(all_qids)

    # 3. Full triple sample
    sample_triples = sample_all_triples(sample_qids, qid_to_label)

    # 4. Physics neighbor discovery
    neighbors = discover_physics_neighbors(sample_qids)

    # 5. Focused predicate exploration
    focused_triples = explore_interesting_predicates(all_qids)

    # ── Save outputs ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    save_json(freq_data, "predicate_frequency.json")
    save_json(inter_links, "inter_entity_links.json")
    save_json(sample_triples, "sample_triples.json")
    save_json(neighbors, "physics_neighbors.json")
    save_json(focused_triples, "focused_predicates.json")

    save_tsv(inter_links, "inter_entity_links.tsv",
             ["entityLabel", "propLabel", "targetLabel", "entity", "prop", "target"])
    save_tsv(sample_triples, "sample_triples.tsv",
             ["entityLabel", "propLabel", "valueLabel", "entity", "prop", "value"])
    save_tsv(focused_triples, "focused_predicates.tsv",
             ["entityLabel", "propLabel", "valueLabel", "entity", "prop", "value"])

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Seed entities queried:     {len(all_qids)}")
    print(f"  Distinct predicates found: {len(freq_data)}")
    print(f"  Inter-entity links:        {len(inter_links)}")
    print(f"  Sample triples:            {len(sample_triples)}")
    print(f"  Physics neighbors:         {len(neighbors)}")
    print(f"  Focused predicate triples: {len(focused_triples)}")

    # Key finding
    if freq_data:
        hier_count = sum(f["count"] for f in freq_data if f["type"] == "HIERARCHY")
        total_count = sum(f["count"] for f in freq_data)
        rich_pct = 100 * (total_count - hier_count) / max(total_count, 1)
        print(f"\n  ★ {rich_pct:.1f}% of predicates are NON-hierarchical")
        print(f"    → Wikidata does encode meaningful physics relationships,")
        print(f"      not just taxonomic hierarchies.")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
