#!/usr/bin/env python3
"""Quick script to validate all seed QIDs against their Wikidata labels."""

from seed_entities import SEED_ENTITIES
from wikidata_sparql import run_sparql, _val, _qid_from_uri
import time

def validate_all():
    qids = list(set(SEED_ENTITIES.values()))
    batch_size = 30
    mismatches = []

    for i in range(0, len(qids), batch_size):
        batch = qids[i:i+batch_size]
        values = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT ?entity ?entityLabel WHERE {{
          VALUES ?entity {{ {values} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        time.sleep(1.5)
        results = run_sparql(query)
        qid_to_wdlabel = {}
        for r in results:
            qid = _qid_from_uri(_val(r, "entity"))
            label = _val(r, "entityLabel")
            qid_to_wdlabel[qid] = label

        for expected_label, qid in SEED_ENTITIES.items():
            if qid in qid_to_wdlabel:
                wd_label = qid_to_wdlabel[qid]
                # Simple check: see if the expected label appears somewhere in the WD label or vice versa
                e = expected_label.lower()
                w = wd_label.lower() if wd_label else ""
                if e not in w and w not in e and e.split()[0] not in w:
                    print(f"  MISMATCH: '{expected_label}' -> {qid} -> Wikidata says: '{wd_label}'")
                    mismatches.append((expected_label, qid, wd_label))

    if not mismatches:
        print("All QIDs match!")
    else:
        print(f"\n{len(mismatches)} mismatches found. Corrections needed:")
        for expected, qid, actual in mismatches:
            print(f"  {expected}: {qid} is actually '{actual}'")

if __name__ == "__main__":
    validate_all()
