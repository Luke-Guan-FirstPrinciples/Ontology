"""
SPARQL query helpers for querying Wikidata's physics knowledge graph.

Uses the public Wikidata Query Service (https://query.wikidata.org/sparql).
All queries are read-only and respect rate limits with a polite User-Agent.
"""

import time
import json
import requests
from typing import Optional

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "PhysicsOntologyExplorer/0.1 (research; mailto:ontology-research@example.com)"

# Polite delay between requests (seconds)
REQUEST_DELAY = 1.5


def run_sparql(query: str, retries: int = 2) -> list[dict]:
    """
    Execute a SPARQL query against Wikidata and return the result bindings.

    Returns a list of dicts, where each dict maps variable names to their values.
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429 and attempt < retries:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(5)
                continue
            raise
    return []


def _val(binding: dict, key: str) -> Optional[str]:
    """Extract value string from a SPARQL result binding."""
    if key in binding:
        return binding[key].get("value")
    return None


def _qid_from_uri(uri: str) -> str:
    """Extract QID or PID from a full Wikidata URI."""
    return uri.rsplit("/", 1)[-1] if uri else uri


# ─────────────────────────────────────────────────────────────────────
# Query 1: Get ALL outgoing triples for a batch of entities
# ─────────────────────────────────────────────────────────────────────

def query_all_outgoing_triples(qids: list[str], limit: int = 2000) -> list[dict]:
    """
    For a list of QIDs, retrieve all (entity, predicate, object) triples
    where the entity is the subject.

    Returns a list of dicts with keys:
      entity, entityLabel, prop, propLabel, value, valueLabel
    """
    values_clause = " ".join(f"wd:{q}" for q in qids)

    query = f"""
    SELECT ?entity ?entityLabel ?prop ?propLabel ?value ?valueLabel
    WHERE {{
      VALUES ?entity {{ {values_clause} }}
      ?entity ?p ?value .

      # Resolve the direct-claim property to its property entity
      ?property wikibase:directClaim ?p .
      BIND(?property AS ?prop)

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    time.sleep(REQUEST_DELAY)
    results = run_sparql(query)

    triples = []
    for r in results:
        triples.append({
            "entity":       _qid_from_uri(_val(r, "entity")),
            "entityLabel":  _val(r, "entityLabel"),
            "prop":         _qid_from_uri(_val(r, "prop")),
            "propLabel":    _val(r, "propLabel"),
            "value":        _val(r, "value"),
            "valueLabel":   _val(r, "valueLabel"),
        })
    return triples


# ─────────────────────────────────────────────────────────────────────
# Query 2: Get triples for specific predicates of interest
# ─────────────────────────────────────────────────────────────────────

def query_specific_predicates(qids: list[str], pids: list[str], limit: int = 1000) -> list[dict]:
    """
    For a list of entity QIDs and predicate PIDs, retrieve matching triples.
    More targeted than query_all_outgoing_triples.
    """
    entity_values = " ".join(f"wd:{q}" for q in qids)
    prop_values = " ".join(f"wdt:{p}" for p in pids)

    query = f"""
    SELECT ?entity ?entityLabel ?prop ?propLabel ?value ?valueLabel
    WHERE {{
      VALUES ?entity {{ {entity_values} }}

      ?entity ?directProp ?value .

      # Only keep predicates we care about
      VALUES ?directProp {{ {prop_values} }}

      # Resolve back to the property entity for labeling
      ?property wikibase:directClaim ?directProp .
      BIND(?property AS ?prop)

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    time.sleep(REQUEST_DELAY)
    results = run_sparql(query)

    triples = []
    for r in results:
        triples.append({
            "entity":       _qid_from_uri(_val(r, "entity")),
            "entityLabel":  _val(r, "entityLabel"),
            "prop":         _qid_from_uri(_val(r, "prop")),
            "propLabel":    _val(r, "propLabel"),
            "value":        _val(r, "value"),
            "valueLabel":   _val(r, "valueLabel"),
        })
    return triples


# ─────────────────────────────────────────────────────────────────────
# Query 3: Predicate frequency analysis
# ─────────────────────────────────────────────────────────────────────

def query_predicate_frequency(qids: list[str], limit: int = 100) -> list[dict]:
    """
    Count how often each predicate appears across the given entities.
    Returns list sorted by frequency descending.
    """
    values_clause = " ".join(f"wd:{q}" for q in qids)

    query = f"""
    SELECT ?prop ?propLabel (COUNT(*) AS ?count)
    WHERE {{
      VALUES ?entity {{ {values_clause} }}
      ?entity ?p ?value .
      ?property wikibase:directClaim ?p .
      BIND(?property AS ?prop)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?prop ?propLabel
    ORDER BY DESC(?count)
    LIMIT {limit}
    """

    time.sleep(REQUEST_DELAY)
    results = run_sparql(query)

    freq = []
    for r in results:
        freq.append({
            "prop":      _qid_from_uri(_val(r, "prop")),
            "propLabel": _val(r, "propLabel"),
            "count":     int(_val(r, "count")),
        })
    return freq


# ─────────────────────────────────────────────────────────────────────
# Query 4: Inter-entity connections (entity→entity only, no literals)
# ─────────────────────────────────────────────────────────────────────

def query_inter_entity_links(qids: list[str], limit: int = 2000) -> list[dict]:
    """
    Find triples where BOTH subject and object are in our seed set.
    These are the most interesting for ontology building—they show
    how physics concepts relate to each other, not just to external values.
    """
    values_clause = " ".join(f"wd:{q}" for q in qids)

    query = f"""
    SELECT ?entity ?entityLabel ?prop ?propLabel ?target ?targetLabel
    WHERE {{
      VALUES ?entity {{ {values_clause} }}
      VALUES ?target {{ {values_clause} }}
      ?entity ?p ?target .
      FILTER(?entity != ?target)

      ?property wikibase:directClaim ?p .
      BIND(?property AS ?prop)

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    time.sleep(REQUEST_DELAY)
    results = run_sparql(query)

    links = []
    for r in results:
        links.append({
            "entity":       _qid_from_uri(_val(r, "entity")),
            "entityLabel":  _val(r, "entityLabel"),
            "prop":         _qid_from_uri(_val(r, "prop")),
            "propLabel":    _val(r, "propLabel"),
            "target":       _qid_from_uri(_val(r, "target")),
            "targetLabel":  _val(r, "targetLabel"),
        })
    return links


# ─────────────────────────────────────────────────────────────────────
# Query 5: Discover physics neighbors (1-hop expansion)
# ─────────────────────────────────────────────────────────────────────

def query_physics_neighbors(qids: list[str], limit: int = 500) -> list[dict]:
    """
    Find entities that are connected to our seed set AND are themselves
    instances or subclasses of physics-related classes.

    Uses a filter to keep only entities in physics-adjacent categories.
    """
    values_clause = " ".join(f"wd:{q}" for q in qids)

    query = f"""
    SELECT DISTINCT ?neighbor ?neighborLabel ?connectingProp ?connectingPropLabel
    WHERE {{
      VALUES ?seed {{ {values_clause} }}

      # Outgoing or incoming link to a neighbor
      {{ ?seed ?p ?neighbor . }}
      UNION
      {{ ?neighbor ?p ?seed . }}

      # Neighbor must be a Wikidata item (not a literal)
      FILTER(ISIRI(?neighbor))
      FILTER(STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/Q"))

      # Neighbor should be related to physics
      ?neighbor wdt:P31/wdt:P279* ?class .
      VALUES ?class {{
        wd:Q11379      # energy
        wd:Q3054889    # physical quantity
        wd:Q4373292    # physical phenomenon
        wd:Q18362      # particle
        wd:Q17444909   # physical theory
        wd:Q131476     # physical law
        wd:Q11348      # equation
        wd:Q332880     # physical constant
        wd:Q413       # physics (field)
      }}

      ?property wikibase:directClaim ?p .
      BIND(?property AS ?connectingProp)

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    time.sleep(REQUEST_DELAY)
    results = run_sparql(query)

    neighbors = []
    for r in results:
        neighbors.append({
            "neighbor":            _qid_from_uri(_val(r, "neighbor")),
            "neighborLabel":       _val(r, "neighborLabel"),
            "connectingProp":      _qid_from_uri(_val(r, "connectingProp")),
            "connectingPropLabel": _val(r, "connectingPropLabel"),
        })
    return neighbors
