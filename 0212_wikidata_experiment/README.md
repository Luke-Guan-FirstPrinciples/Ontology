# Wikidata Physics Knowledge Graph Exploration

**Date:** 2026-02-12  
**Question:** Does Wikidata contain meaningful relationships between physics entities, or just hierarchies?

## Key Finding

**92.5% of predicates connecting physics entities are non-hierarchical.** Wikidata encodes rich semantic relationships for physics, not just `instance of` / `subclass of` taxonomies.

## What's Here

| File | Purpose |
|------|---------|
| `explore_physics_kg.py` | Main exploration script — run this |
| `seed_entities.py` | 58 curated physics entities with validated Wikidata QIDs |
| `wikidata_sparql.py` | SPARQL query helpers (5 query types) |
| `validate_qids.py` | Utility to validate QIDs against Wikidata labels |

## Results Summary (58 seed entities)

| Metric | Count |
|--------|-------|
| Distinct predicates found | 294 |
| Inter-entity links (physics↔physics) | 79 |
| Sample triples (10 entities) | 807 |
| Physics neighbors (1-hop) | 27 |
| Focused predicate triples | 219 |

## Most Interesting Predicates for Physics Ontology

### Semantically Rich (non-hierarchical) predicates found between physics entities:

| Predicate | Example | Count |
|-----------|---------|-------|
| **interaction** | electron →[interaction]→ gravity | 32 |
| **has characteristic** | photon →[has characteristic]→ wave-particle duality | 7 |
| **studied by** | gravity →[studied by]→ general relativity | 6 |
| **has part(s)** | proton →[has part(s)]→ up quark | 6 |
| **is the study of** | thermodynamics →[is the study of]→ entropy | 3 |
| **facet of** | speed of light →[facet of]→ special relativity | 3 |
| **different from** | neutrino →[different from]→ neutron | 3 |
| **measured physical quantity** | Boltzmann constant →[measured physical quantity]→ entropy | 2 |
| **decays to** | Higgs boson →[decays to]→ photon | 2 |
| **antiparticle** | electron →[antiparticle]→ positron | 2 |
| **has effect** | Big Bang →[has effect]→ cosmic microwave background | 1 |
| **has cause** | Hawking radiation →[has cause]→ black hole | 1 |

### Notable focused predicate findings:

- **`has part(s)`** encodes compositional physics: proton has parts {gluon, up quark, down quark}, Maxwell's equations has parts {Ampere's law, Gauss's law, Faraday's law, Gauss's law for magnetism}
- **`has characteristic`** connects entities to their physical properties: black hole has {electric charge, mass, angular momentum}, quark has {electric charge, color charge}
- **`discoverer or inventor`** links entities to scientists: electron → J.J. Thomson, general relativity → Albert Einstein
- **`facet of`** and **`studied by`** encode disciplinary relationships: quantum entanglement studied by quantum mechanics, Big Bang facet of general relativity

## How to Expand

1. **More entities:** Add QIDs to `SEED_ENTITIES` in `seed_entities.py`
2. **More predicates:** Add PIDs to `PHYSICS_PREDICATES_OF_INTEREST`  
3. **Multi-hop expansion:** Use the neighbor discovery (Step 4) output as new seeds
4. **Build a graph:** Load the JSON/TSV output into NetworkX, Neo4j, or similar

## Running

```bash
pip install requests
python explore_physics_kg.py
```

Queries the public Wikidata SPARQL endpoint. No API key needed. Rate-limited to ~1.5s between queries.
