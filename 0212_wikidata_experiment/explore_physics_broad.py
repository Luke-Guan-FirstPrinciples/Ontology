#!/usr/bin/env python3
"""
Wikidata Physics Knowledge Graph — Broad Exploration (No Seed Constraint)
=========================================================================

Key insight about Wikidata's data model for physics:
  - PARTICLES are modeled as P279 (subclass of) hierarchies.
    "electron" is a subclass of "elementary particle", not an instance.
    Because individual electrons aren't Wikidata items — "electron" IS the type.
  - THEORIES, LAWS, CONSTANTS, PHENOMENA are P31 (instance of).
    "general relativity" is an instance of "physical law".

This script handles both patterns correctly.

Usage:
  python explore_physics_broad.py
"""

import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

from wikidata_sparql import run_sparql, _val, _qid_from_uri

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_broad")
DELAY = 2.5  # seconds between queries

# ─────────────────────────────────────────────────────────────────────
# Physics classes to crawl
# (QID, label, strategy)
#   strategy = "subclass"  → P279* (particles — they ARE subclasses)
#   strategy = "instance"  → P31/P279* (theories, laws — they are instances)
# ─────────────────────────────────────────────────────────────────────

PHYSICS_CLASSES = [
    # ── Particles (subclass hierarchies) ──────────────────────────────
    ("Q43116",    "elementary particle",     "subclass"),
    ("Q43101",    "boson",                   "subclass"),
    ("Q44363",    "fermion",                 "subclass"),
    ("Q101667",   "hadron",                  "subclass"),
    ("Q159731",   "baryon",                  "subclass"),
    ("Q102742",   "meson",                   "subclass"),
    ("Q82586",    "lepton",                  "subclass"),
    ("Q24364",    "force carrier",           "subclass"),

    # ── Physical quantities (instance hierarchies) ────────────────────
    ("Q107715",   "physical quantity",       "instance"),
    ("Q30337748", "ISQ base quantity",       "instance"),

    # ── Constants ─────────────────────────────────────────────────────
    ("Q173227",   "physical constant",       "instance"),

    # ── Theories ──────────────────────────────────────────────────────
    ("Q9357058",  "physical theory",         "instance"),
    ("Q17444909", "astronomical object type","instance"),  # includes black hole types etc.

    # ── Laws ──────────────────────────────────────────────────────────
    ("Q214070",   "physical law",            "instance"),

    # ── Phenomena ─────────────────────────────────────────────────────
    ("Q4373292",  "physical phenomenon",     "instance"),
    ("Q3457198",  "macroscopic quantum phenomena", "instance"),

    # ── Nuclear ───────────────────────────────────────────────────────
    ("Q238323",   "nuclear reaction",        "instance"),
]

# ─────────────────────────────────────────────────────────────────────
# Semantic predicates
# ─────────────────────────────────────────────────────────────────────

SEMANTIC_PREDICATES = [
    # Ontological
    "P31",    # instance of
    "P279",   # subclass of
    "P361",   # part of
    "P527",   # has part(s)
    "P1269",  # facet of

    # Relational
    "P1552",  # has characteristic
    "P1889",  # different from
    "P460",   # said to be the same as
    "P461",   # opposite of
    "P2283",  # uses
    "P1535",  # used by
    "P737",   # influenced by
    "P144",   # based on
    "P828",   # has cause
    "P1542",  # has effect
    "P1534",  # end cause

    # Particle physics
    "P517",   # interaction
    "P4354",  # decays to
    "P111",   # measured physical quantity
    "P3403",  # antiparticle

    # Discovery
    "P61",    # discoverer or inventor
    "P575",   # time of discovery or invention
    "P138",   # named after

    # Physical values
    "P2067",  # mass
    "P2200",  # electric charge
    "P1109",  # spin quantum number

    # Disciplinary
    "P2579",  # studied in
    "P2578",  # is the study of
    "P366",   # has use

    # Formulas
    "P2534",  # defining formula
    "P7235",  # in defining formula

    # Temporal
    "P155",   # follows
    "P156",   # followed by
]


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Discover entities
# ─────────────────────────────────────────────────────────────────────

def discover_entities(class_qid: str, strategy: str) -> list:
    if strategy == "subclass":
        path = "wdt:P279*"
    else:
        path = "wdt:P31/wdt:P279*"

    query = f"""
    SELECT DISTINCT ?entity ?entityLabel ?entityDescription
    WHERE {{
      ?entity {path} wd:{class_qid} .
      FILTER(STRSTARTS(STR(?entity), "http://www.wikidata.org/entity/Q"))
      FILTER NOT EXISTS {{ ?entity wdt:P31 wd:Q4167410 . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5000
    """
    time.sleep(DELAY)
    try:
        return run_sparql(query)
    except Exception as e:
        print(f" ERROR: {e}")
        return []


def discover_all() -> dict:
    print("\n" + "=" * 70)
    print("PHASE 1: Discovering physics entities")
    print("=" * 70)

    pool = {}

    for class_qid, class_label, strategy in PHYSICS_CLASSES:
        print(f"  {class_label} ({class_qid}, {strategy})...", end="", flush=True)
        results = discover_entities(class_qid, strategy)
        count = 0
        for r in results:
            qid = _qid_from_uri(_val(r, "entity"))
            if not (qid and qid.startswith("Q")):
                continue
            count += 1
            if qid not in pool:
                pool[qid] = {
                    "label": _val(r, "entityLabel"),
                    "description": _val(r, "entityDescription"),
                    "classes": [class_label],
                }
            else:
                if class_label not in pool[qid]["classes"]:
                    pool[qid]["classes"].append(class_label)
        print(f" → {count}")

    print(f"\n  ═══ Total unique entities: {len(pool)} ═══\n")
    cc = Counter()
    for info in pool.values():
        for c in info["classes"]:
            cc[c] += 1
    for cls, cnt in cc.most_common():
        print(f"    {cls:<45} {cnt}")

    return pool


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Get semantic triples (batched by entity QIDs)
# ─────────────────────────────────────────────────────────────────────

def query_triples_batch(qids: list[str], pred_pids: list[str]) -> list[dict]:
    """Query semantic triples for a batch of entities (by QID VALUES clause)."""
    entity_values = " ".join(f"wd:{q}" for q in qids)
    prop_values = " ".join(f"wdt:{p}" for p in pred_pids)

    query = f"""
    SELECT ?entity ?entityLabel ?prop ?propLabel ?value ?valueLabel
    WHERE {{
      VALUES ?entity {{ {entity_values} }}
      VALUES ?directProp {{ {prop_values} }}
      ?entity ?directProp ?value .
      ?property wikibase:directClaim ?directProp .
      BIND(?property AS ?prop)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10000
    """

    time.sleep(DELAY)
    try:
        results = run_sparql(query)
    except Exception as e:
        # If batch is too large, split in half and retry
        if len(qids) > 10:
            mid = len(qids) // 2
            r1 = query_triples_batch(qids[:mid], pred_pids)
            r2 = query_triples_batch(qids[mid:], pred_pids)
            return r1 + r2
        print(f" ERROR: {e}")
        return []

    triples = []
    for r in results:
        val_raw = _val(r, "value") or ""
        if "commons.wikimedia.org" in val_raw:
            continue

        triples.append({
            "entity":       _qid_from_uri(_val(r, "entity")),
            "entityLabel":  _val(r, "entityLabel"),
            "prop":         _qid_from_uri(_val(r, "prop")),
            "propLabel":    _val(r, "propLabel"),
            "value":        val_raw,
            "valueLabel":   _val(r, "valueLabel") or "",
        })
    return triples


def collect_all_triples(pool: dict) -> list[dict]:
    print("\n" + "=" * 70)
    print("PHASE 2: Collecting semantic triples")
    print("=" * 70)

    all_qids = list(pool.keys())
    batch_size = 50  # entities per SPARQL query
    all_triples = []
    seen = set()

    batches = list(batched(all_qids, batch_size))
    print(f"  {len(all_qids)} entities in {len(batches)} batches of {batch_size}")

    for i, batch in enumerate(batches, 1):
        triples = query_triples_batch(batch, SEMANTIC_PREDICATES)
        new = 0
        for t in triples:
            key = (t["entity"], t["prop"], t["value"])
            if key not in seen:
                seen.add(key)
                all_triples.append(t)
                new += 1
        if i % 5 == 0 or i == len(batches):
            print(f"  Batch {i}/{len(batches)}: +{new} triples (total: {len(all_triples)})")

    print(f"\n  ═══ Total semantic triples: {len(all_triples)} ═══")
    return all_triples


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Inter-entity links
# ─────────────────────────────────────────────────────────────────────

def build_inter_entity_links(triples: list[dict], pool: dict) -> list[dict]:
    print("\n" + "=" * 70)
    print("PHASE 3: Inter-Entity Links")
    print("=" * 70)

    known = set(pool.keys())
    links = []

    for t in triples:
        val = t["value"]
        if "wikidata.org/entity/Q" in val:
            tgt = val.rsplit("/", 1)[-1]
            if tgt in known and tgt != t["entity"]:
                links.append({
                    "entity": t["entity"], "entityLabel": t["entityLabel"],
                    "prop": t["prop"], "propLabel": t["propLabel"],
                    "target": tgt, "targetLabel": pool[tgt]["label"],
                })

    hier_pids = {"P31", "P279"}
    hier = [l for l in links if l["prop"] in hier_pids]
    rich = [l for l in links if l["prop"] not in hier_pids]

    print(f"  Total: {len(links)}  (hierarchical: {len(hier)}, semantic: {len(rich)})")

    pc = Counter(l["propLabel"] for l in rich)
    if pc:
        print(f"\n  ── Non-hierarchical predicate breakdown ──")
        for pred, count in pc.most_common(30):
            print(f"    {pred:<50} {count}")

    by_pred = defaultdict(list)
    for l in rich:
        by_pred[l["propLabel"]].append(l)

    print(f"\n  ── Samples ──")
    for pred_label, pl in sorted(by_pred.items(), key=lambda x: -len(x[1])):
        print(f"\n  {pred_label} ({len(pl)}):")
        for l in pl[:8]:
            print(f"    {l['entityLabel']:<40} → {l['targetLabel']}")
        if len(pl) > 8:
            print(f"    ... +{len(pl)-8} more")

    return links


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Analysis
# ─────────────────────────────────────────────────────────────────────

def analyze(triples, links, pool):
    print("\n" + "=" * 70)
    print("PHASE 4: Analysis")
    print("=" * 70)

    # Predicate freq
    pc = Counter()
    pl = {}
    for t in triples:
        pc[t["prop"]] += 1
        pl[t["prop"]] = t["propLabel"]
    sorted_p = sorted(pc.items(), key=lambda x: -x[1])

    print(f"\n  ── Predicate frequency ({len(sorted_p)} predicates) ──")
    print(f"  {'#':<4} {'PID':<8} {'Label':<50} {'Count'}")
    print("  " + "-" * 75)
    freq = []
    for i, (pid, count) in enumerate(sorted_p, 1):
        label = pl.get(pid, "?")
        print(f"  {i:<4} {pid:<8} {label:<50} {count}")
        freq.append({"rank": i, "pid": pid, "label": label, "count": count})

    # Connectivity
    conn = Counter()
    for l in links:
        conn[l["entity"]] += 1
        conn[l["target"]] += 1

    print(f"\n  ── Connectivity: {len(conn)}/{len(pool)} entities connected ──")
    print(f"\n  Top 50:")
    for qid, count in conn.most_common(50):
        info = pool.get(qid, {})
        label = info.get("label", qid)
        classes = ", ".join(info.get("classes", [])[:3])
        print(f"    {label:<45} {count:>4}  [{classes}]")

    return freq, conn


# ─────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────

def save_json(data, filename):
    fp = os.path.join(OUTPUT_DIR, filename)
    with open(fp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    mb = os.path.getsize(fp) / 1048576
    print(f"  {fp} ({mb:.1f} MB, {len(data)} records)")


def save_tsv(rows, filename, cols):
    fp = os.path.join(OUTPUT_DIR, filename)
    with open(fp, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    mb = os.path.getsize(fp) / 1048576
    print(f"  {fp} ({mb:.1f} MB, {len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Wikidata Physics KG — Broad Exploration (No Seed Constraint)    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print(f"  Classes:    {len(PHYSICS_CLASSES)}")
    print(f"  Predicates: {len(SEMANTIC_PREDICATES)}")

    ensure_output_dir()

    pool = discover_all()
    triples = collect_all_triples(pool)
    links = build_inter_entity_links(triples, pool)
    freq, conn = analyze(triples, links, pool)

    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    catalog = [{"qid": q, "label": i["label"], "description": i["description"],
                "classes": i["classes"]}
               for q, i in sorted(pool.items(), key=lambda x: (x[1]["label"] or "").lower())]

    save_json(catalog, "entity_catalog.json")
    save_tsv([{"qid": e["qid"], "label": e["label"], "description": e["description"],
               "classes": "; ".join(e["classes"])} for e in catalog],
             "entity_catalog.tsv", ["qid", "label", "description", "classes"])
    save_json(triples, "semantic_triples.json")
    save_tsv(triples, "semantic_triples.tsv",
             ["entityLabel", "propLabel", "valueLabel", "entity", "prop", "value"])
    save_json(links, "inter_entity_links.json")
    save_tsv(links, "inter_entity_links.tsv",
             ["entityLabel", "propLabel", "targetLabel", "entity", "prop", "target"])
    save_json(freq, "predicate_frequency.json")

    elapsed = time.time() - t0
    hier = sum(1 for l in links if l["prop"] in {"P31", "P279"})

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Unique physics entities:    {len(pool)}")
    print(f"  Semantic triples:           {len(triples)}")
    print(f"  Distinct predicates:        {len(freq)}")
    print(f"  Inter-entity links:         {len(links)}")
    print(f"    Hierarchical:             {hier}")
    print(f"    Semantically rich:        {len(links) - hier}")
    print(f"  Connected entities:         {len(conn)} / {len(pool)}")
    print(f"  Runtime:                    {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"\n  Output: {OUTPUT_DIR}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
