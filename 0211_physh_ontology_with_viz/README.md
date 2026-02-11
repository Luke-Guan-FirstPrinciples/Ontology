# Physics Ontology Builder

This directory contains scripts to build a Physics Ontology (PhySH-based) leveraging APS articles for semantic edge induction.

## Components

1. **`config.yaml`**: Configuration for database, facet mappings, and thresholds.
2. **`ontology_builder.py`**:
   - Loads PhySH concepts.
   - Scans `aps_articles` to find co-occurrences of concepts.
   - Induces semantic edges (e.g., Area->System, System->Property) based on configurable rules.
   - Populates `third_party_aps.ontology_edges` with induced edges (including support/weight).
   - Populates `third_party_aps.ontology_paper_evidence` with Paper->Concept links.
3. **`extract_structural_edges.py`**:
   - Extracts structural edges (HAS_FACET, IN_DISCIPLINE, PARENT_OF, RELATED_TO) from the existing taxonomy.
   - Upserts them into `third_party_aps.ontology_edges`.

## Usage

1. **Build Semantic Ontology**:
   ```bash
   python3 ontology_builder.py
   ```
   *Note: This truncates `ontology_edges` and `ontology_paper_evidence` to rebuild them from scratch.*

2. **Add Structural Edges**:
   ```bash
   python3 extract_structural_edges.py
   ```
   *Run this AFTER `ontology_builder.py` to ensure structural edges are added to the table.*

## Schema

- **`ontology_edges`**:
  - `source_id` (UUID)
  - `target_id` (UUID)
  - `relation` (VARCHAR): e.g., STUDIES, EXHIBITS, MEASURES, HAS_FACET.
  - `weight` (FLOAT): 0.5-1.0 for induced, 1.0 for structural.
  - `support` (INT): Number of papers supporting the edge (0 for structural).

- **`ontology_paper_evidence`**:
  - `doi` (VARCHAR)
  - `concept_id` (UUID)
  - `relation` (VARCHAR): MENTIONS or USES.
  - `is_primary` (BOOLEAN)
