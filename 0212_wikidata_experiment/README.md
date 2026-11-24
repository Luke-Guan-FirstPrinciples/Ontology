# Wikidata Physics Knowledge Graph Exploration

**Date:** 2026-02-12  
**Question:** Does Wikidata contain meaningful relationships between physics entities, or just hierarchies?

## Two Explorations

### 1. Seed-Based (58 hand-picked entities) → `output/`

Quick, targeted exploration starting from 58 curated entities (electron, photon, Standard Model, etc.). Good for understanding the data model.

```bash
python explore_physics_kg.py
```

### 2. Broad Exploration (no seed constraint) → `output_broad/`

Crawls 17 Wikidata physics classes to discover ALL physics entities and relationships. No hand-picking.

```bash
python explore_physics_broad.py
```

## Broad Exploration Results

| Metric | Count |
|--------|-------|
| Physics classes crawled | 17 |
| **Unique physics entities** | **8,964** |
| **Semantic triples (clean)** | **33,062** |
| Distinct predicates | 30 |
| **Inter-entity links** | **9,682** |
| — Hierarchical (instance of / subclass of) | 3,408 |
| — **Semantically rich** | **6,274** |
| Connected entities | 4,607 / 8,964 (51%) |

### Entity breakdown by class

| Class | Entities |
|-------|----------|
| physical quantity | 5,000 (hit LIMIT) |
| physical phenomenon | 5,000 (hit LIMIT) |
| astronomical object type | 438 |
| physical law | 228 |
| physical theory | 195 |
| boson | 120 |
| fermion | 119 |
| elementary particle | 113 |
| hadron | 100 |
| physical constant | 95 |
| meson | 42 |
| baryon | 35 |
| lepton | 30 |
| force carrier | 23 |
| nuclear reaction | 10 |

### Top semantic predicates (all 30 are meaningful — noise was filtered)

| # | Predicate | Triples |
|---|-----------|---------|
| 1 | instance of | 11,234 |
| 2 | has part(s) | 5,231 |
| 3 | part of | 4,234 |
| 4 | followed by | 2,995 |
| 5 | follows | 2,987 |
| 6 | subclass of | 1,618 |
| 7 | different from | 904 |
| 8 | measured physical quantity | 731 |
| 9 | named after | 670 |
| 10 | in defining formula | 490 |
| 11 | said to be the same as | 335 |
| 12 | interaction | 272 |
| 13 | defining formula | 261 |
| 14 | facet of | 222 |
| 15 | discoverer or inventor | 134 |

### Non-hierarchical inter-entity links (6,274 total)

| Predicate | Links | Example |
|-----------|-------|---------|
| followed by / follows | 5,112 | muon antineutrino → tau antineutrino |
| measured physical quantity | 407 | Boltzmann constant → entropy |
| different from | 212 | W boson → ω-meson |
| has part(s) | 198 | ω-meson → {up quark, down quark} |
| part of | 182 | quark → hadron |
| said to be the same as | 64 | speed of light → speed of gravity |
| opposite of | 42 | boson → fermion; electron → proton |
| named after | 21 | hexaquark → quark |
| facet of | 13 | speed of light → special relativity |
| has characteristic | 11 | photon → wave-particle duality |

## Key Wikidata Data Model Insight

Physics entities use **two different patterns** in Wikidata:

- **Particles** are `P279 (subclass of)` hierarchies. Electron is a *subclass* of elementary particle, not an instance. Because individual electrons aren't Wikidata items — "electron" is the type itself.
- **Theories, laws, constants, phenomena** are `P31 (instance of)`. General relativity is an *instance* of "physical law".

This matters for querying. The broad script handles both correctly.

## File Structure

| File | Purpose |
|------|---------|
| `explore_physics_broad.py` | **Broad exploration** — discovers all physics entities by class |
| `explore_physics_kg.py` | Seed-based exploration (58 entities) |
| `seed_entities.py` | 58 curated physics entities with validated QIDs |
| `wikidata_sparql.py` | SPARQL query helpers |
| `validate_qids.py` | QID validation utility |
| `output/` | Seed-based results |
| `output_broad/` | Broad exploration results |

## Output Files (output_broad/)

| File | Size | Records |
|------|------|---------|
| `entity_catalog.json` | 1.4 MB | 8,964 entities |
| `semantic_triples.json` | 7.2 MB | 33,062 triples |
| `inter_entity_links.json` | 1.5 MB | 9,682 links |
| `predicate_frequency.json` | — | 30 predicates |
| TSV versions of all the above | | |

## Running

```bash
pip install requests
python explore_physics_broad.py   # ~11 minutes
python explore_physics_kg.py      # ~1 minute (seed-based)
```

Queries the public Wikidata SPARQL endpoint. No API key needed. Rate-limited to ~2.5s between queries.
