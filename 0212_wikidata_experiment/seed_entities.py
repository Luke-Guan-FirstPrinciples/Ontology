"""
Seed physics entities for Wikidata exploration.

Each entry maps a human-readable label to its Wikidata QID.
Organized by ontological category to make the structure explicit.

QIDs validated against Wikidata on 2026-02-12.
"""

SEED_ENTITIES = {
    # ── Fundamental Particles ──────────────────────────────────────────
    "electron":             "Q2225",
    "photon":               "Q3198",
    "proton":               "Q2294",
    "neutron":              "Q2348",
    "neutrino":             "Q2126",
    "muon":                 "Q3151",
    "Higgs boson":          "Q402",
    "W boson":              "Q1029485",
    "Z boson":              "Q488719",
    "gluon":                "Q3299",
    "quark":                "Q6718",
    "up quark":             "Q6732",
    "down quark":           "Q6745",
    "positron":             "Q3229",

    # ── Fundamental Forces / Interactions ──────────────────────────────
    "electromagnetism":     "Q11406",
    "gravity":              "Q11412",
    "strong interaction":   "Q11415",
    "weak interaction":     "Q11418",

    # ── Theories & Frameworks ─────────────────────────────────────────
    "Standard Model":               "Q18338",
    "quantum mechanics":            "Q944",
    "general relativity":           "Q11452",
    "special relativity":           "Q11455",
    "quantum field theory":         "Q54505",
    "quantum electrodynamics":      "Q234881",
    "quantum chromodynamics":       "Q238170",
    "string theory":                "Q33198",
    "thermodynamics":               "Q11473",
    "statistical mechanics":        "Q188715",
    "classical mechanics":          "Q11397",

    # ── Equations & Laws ──────────────────────────────────────────────
    "Schrödinger equation":         "Q165498",
    "Maxwell's equations":          "Q51501",
    "Newton's laws of motion":      "Q38433",
    "Dirac equation":               "Q272621",
    "Einstein field equations":     "Q273711",
    "Navier–Stokes equations":      "Q201321",
    "uncertainty principle":        "Q44746",

    # ── Physical Constants ────────────────────────────────────────────
    "speed of light":               "Q2111",
    "Planck constant":              "Q122894",
    "gravitational constant":       "Q30006",
    "Boltzmann constant":           "Q5962",
    "elementary charge":            "Q2101",
    "fine-structure constant":      "Q5997",

    # ── Physical Properties / Quantities ──────────────────────────────
    "mass":                 "Q11423",
    "electric charge":      "Q1111",
    "spin quantum number":  "Q3879445",
    "momentum":             "Q41273",
    "energy":               "Q11379",
    "wavelength":           "Q41364",
    "angular momentum":     "Q161254",
    "entropy":              "Q45003",
    "temperature":          "Q11466",

    # ── Phenomena ─────────────────────────────────────────────────────
    "superconductivity":    "Q124131",
    "black hole":           "Q589",
    "Big Bang":             "Q323",
    "quantum entanglement": "Q215675",
    "wave–particle duality":"Q193068",
    "Bose–Einstein condensate": "Q46202",
    "Hawking radiation":    "Q497396",
}

# Also useful: well-known physics predicates on Wikidata
# These are the property IDs (PIDs) we'll look for.
PHYSICS_PREDICATES_OF_INTEREST = {
    # ── Ontological / Taxonomic ───────────────────────────────────────
    "P31":   "instance of",
    "P279":  "subclass of",
    "P361":  "part of",
    "P527":  "has part(s)",
    "P1269": "facet of",

    # ── Descriptive / Relational ──────────────────────────────────────
    "P1552": "has characteristic",
    "P1889": "different from",
    "P460":  "said to be the same as",
    "P2283": "uses",
    "P1535": "used by",
    "P737":  "influenced by",
    "P144":  "based on",
    "P828":  "has cause",
    "P1542": "has effect",

    # ── Discovery & History ───────────────────────────────────────────
    "P61":   "discoverer or inventor",
    "P575":  "time of discovery or invention",
    "P138":  "named after",

    # ── Physical Properties (quantity-valued) ─────────────────────────
    "P2067": "mass",
    "P2200": "electric charge",
    "P1109": "spin quantum number",
    "P2354": "has average lifetime",

    # ── Scientific Context ────────────────────────────────────────────
    "P101":  "field of this occupation",
    "P2579": "studied in",
    "P910":  "topic's main category",
    "P1269": "facet of",
}
