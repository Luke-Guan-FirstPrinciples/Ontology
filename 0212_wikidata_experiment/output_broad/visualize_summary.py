#!/usr/bin/env python3
"""
Wikidata Physics Knowledge Graph — Summary & Visualizations
============================================================
Generates a comprehensive visual report from the output_broad/ data.

Outputs:
  - summary_report.txt          — text summary of key metrics
  - fig1_predicate_frequency.png — bar chart of predicate frequencies
  - fig2_entity_classes.png      — treemap of entity class distribution
  - fig3_link_types.png          — stacked bar: hierarchical vs semantic links
  - fig4_network_sample.png      — network graph of top inter-entity links
  - fig5_class_connectivity.png  — heatmap: which classes link to which
  - fig6_top_entities.png        — top 25 most-connected entities
  - fig7_predicate_categories.png — predicate grouping donut chart
"""

import json
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR / "visualizations"
OUT_DIR.mkdir(exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 30)
DPI = 180

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data...")
with open(DATA_DIR / "entity_catalog.json") as f:
    entities = json.load(f)
with open(DATA_DIR / "semantic_triples.json") as f:
    triples = json.load(f)
with open(DATA_DIR / "inter_entity_links.json") as f:
    links = json.load(f)
with open(DATA_DIR / "predicate_frequency.json") as f:
    pred_freq = json.load(f)

# Build entity lookup
entity_map = {e["qid"]: e for e in entities}

# ── Computed Stats ────────────────────────────────────────────────────────────
print("Computing statistics...")

# Entity class counts
class_counts = Counter()
for e in entities:
    for c in e.get("classes", []):
        class_counts[c] += 1

# Predicate counts from triples
pred_counts = Counter()
for t in triples:
    pred_counts[t.get("propLabel", "unknown")] += 1

# Inter-entity link analysis
link_pred_counts = Counter()
for lnk in links:
    link_pred_counts[lnk.get("propLabel", "unknown")] += 1

hierarchical_preds = {"instance of", "subclass of"}
hier_links = sum(c for p, c in link_pred_counts.items() if p in hierarchical_preds)
semantic_links = sum(c for p, c in link_pred_counts.items() if p not in hierarchical_preds)

# Degree centrality
degree = Counter()
for lnk in links:
    degree[lnk["entity"]] += 1
    degree[lnk["target"]] += 1

# Entities with descriptions
with_desc = sum(1 for e in entities if e.get("description"))

# ── 0. Summary Report ────────────────────────────────────────────────────────
print("Writing summary report...")
report = f"""\
╔══════════════════════════════════════════════════════════════════╗
║       WIKIDATA PHYSICS KNOWLEDGE GRAPH — SUMMARY REPORT        ║
║                    Broad Exploration Results                     ║
╚══════════════════════════════════════════════════════════════════╝

OVERVIEW
────────────────────────────────────────────────────────────────────
  Total unique entities:          {len(entities):>8,}
  Entity classes crawled:         {len(class_counts):>8}
  Entities with descriptions:     {with_desc:>8,} ({100*with_desc/len(entities):.1f}%)
  
  Total semantic triples:         {len(triples):>8,}
  Distinct predicates:            {len(pred_counts):>8}
  
  Inter-entity links:             {len(links):>8,}
    — Hierarchical (P31/P279):    {hier_links:>8,}
    — Semantically rich:          {semantic_links:>8,}
  Connected entities:             {len(degree):>8,} / {len(entities):,} ({100*len(degree)/len(entities):.1f}%)

ENTITY CLASSES (top 17)
────────────────────────────────────────────────────────────────────
"""
for cls, cnt in class_counts.most_common():
    bar = "█" * max(1, int(50 * cnt / class_counts.most_common(1)[0][1]))
    report += f"  {cls:<35} {cnt:>6,}  {bar}\n"

report += f"""
TOP 15 PREDICATES (by triple count)
────────────────────────────────────────────────────────────────────
"""
for p in pred_freq[:15]:
    bar = "█" * max(1, int(50 * p["count"] / pred_freq[0]["count"]))
    report += f"  {p['label']:<35} {p['count']:>6,}  {bar}\n"

report += f"""
TOP 25 MOST-CONNECTED ENTITIES
────────────────────────────────────────────────────────────────────
"""
for qid, deg in degree.most_common(25):
    e = entity_map.get(qid, {})
    label = e.get("label", qid)
    classes = ", ".join(e.get("classes", []))
    report += f"  {label:<35} {deg:>4} links  ({classes})\n"

report += f"""
INTER-ENTITY LINK BREAKDOWN
────────────────────────────────────────────────────────────────────
"""
for pred, cnt in link_pred_counts.most_common():
    kind = "HIER" if pred in hierarchical_preds else "SEM "
    report += f"  [{kind}] {pred:<35} {cnt:>5,}\n"

report += """
════════════════════════════════════════════════════════════════════
"""

report_path = OUT_DIR / "summary_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"  → {report_path}")


# ── 1. Predicate Frequency Bar Chart ─────────────────────────────────────────
print("Generating fig1: Predicate frequency...")
fig, ax = plt.subplots(figsize=(14, 8))
df_pred = pd.DataFrame(pred_freq)
colors = sns.color_palette("viridis", len(df_pred))
bars = ax.barh(
    df_pred["label"][::-1],
    df_pred["count"][::-1],
    color=colors[::-1],
    edgecolor="white",
    linewidth=0.5,
)
ax.set_xlabel("Number of Triples", fontsize=13)
ax.set_title("Predicate Frequency in Semantic Triples", fontsize=16, fontweight="bold", pad=15)
for bar, val in zip(bars, df_pred["count"][::-1]):
    ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9, color="#333")
ax.set_xlim(0, df_pred["count"].max() * 1.15)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_predicate_frequency.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig1_predicate_frequency.png")


# ── 2. Entity Class Distribution (Treemap-style bar) ─────────────────────────
print("Generating fig2: Entity class distribution...")
fig, ax = plt.subplots(figsize=(14, 7))
class_df = pd.DataFrame(
    [(cls, cnt) for cls, cnt in class_counts.most_common()],
    columns=["class", "count"]
)
colors2 = sns.color_palette("mako", len(class_df))
bars = ax.bar(
    range(len(class_df)),
    class_df["count"],
    color=colors2,
    edgecolor="white",
    linewidth=0.5,
)
ax.set_xticks(range(len(class_df)))
ax.set_xticklabels(class_df["class"], rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Number of Entities", fontsize=13)
ax.set_title("Entity Distribution by Physics Class", fontsize=16, fontweight="bold", pad=15)
for bar, val in zip(bars, class_df["count"]):
    if val > 100:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
# Add note about LIMIT
ax.annotate("← hit SPARQL 5,000 LIMIT", xy=(0.5, 5000), fontsize=9,
            color="red", alpha=0.7, xytext=(3, 4800),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))
plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_entity_classes.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig2_entity_classes.png")


# ── 3. Hierarchical vs Semantic Links ────────────────────────────────────────
print("Generating fig3: Link type breakdown...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.5]})

# Left: donut chart
ax = axes[0]
sizes = [hier_links, semantic_links]
labels = [f"Hierarchical\n({hier_links:,})", f"Semantic\n({semantic_links:,})"]
colors3 = ["#4e79a7", "#e15759"]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors3, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75, textprops={"fontsize": 12},
    wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2}
)
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title("Link Type Split", fontsize=14, fontweight="bold")

# Right: breakdown of semantic link types
ax2 = axes[1]
sem_preds = [(p, c) for p, c in link_pred_counts.most_common() if p not in hierarchical_preds]
sem_df = pd.DataFrame(sem_preds, columns=["predicate", "count"])
colors_sem = sns.color_palette("flare", len(sem_df))
bars = ax2.barh(sem_df["predicate"][::-1], sem_df["count"][::-1],
                color=colors_sem[::-1], edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Number of Links", fontsize=12)
ax2.set_title("Non-Hierarchical Inter-Entity Links", fontsize=14, fontweight="bold")
for bar, val in zip(bars, sem_df["count"][::-1]):
    ax2.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=9, color="#333")
ax2.set_xlim(0, sem_df["count"].max() * 1.2)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_link_types.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig3_link_types.png")


# ── 4. Network Graph (sampled) ───────────────────────────────────────────────
print("Generating fig4: Network graph...")
# Build a graph from inter-entity links, focusing on top-connected nodes
G = nx.Graph()
for lnk in links:
    src = lnk["entityLabel"]
    tgt = lnk["targetLabel"]
    prop = lnk["propLabel"]
    if prop in hierarchical_preds:
        continue  # skip hierarchical for cleaner viz
    G.add_edge(src, tgt, label=prop)

# Keep only nodes with degree >= 4 for readability
nodes_to_keep = [n for n, d in G.degree() if d >= 4]
G_sub = G.subgraph(nodes_to_keep).copy()

# If still too big, take the largest connected component
if len(G_sub) > 150:
    largest_cc = max(nx.connected_components(G_sub), key=len)
    G_sub = G_sub.subgraph(largest_cc).copy()

fig, ax = plt.subplots(figsize=(20, 16))
pos = nx.spring_layout(G_sub, k=2.5, iterations=80, seed=42)

# Color by degree
degrees = dict(G_sub.degree())
node_colors = [degrees[n] for n in G_sub.nodes()]
node_sizes = [max(80, degrees[n] * 30) for n in G_sub.nodes()]

# Draw edges
edge_colors_map = {
    "followed by": "#b0b0b0", "follows": "#b0b0b0",
    "has part(s)": "#4e79a7", "part of": "#4e79a7",
    "measured physical quantity": "#e15759",
    "different from": "#f28e2c",
    "said to be the same as": "#76b7b2",
    "opposite of": "#ff9da7",
    "named after": "#9c755f",
}
edge_colors = [edge_colors_map.get(G_sub[u][v].get("label", ""), "#cccccc")
               for u, v in G_sub.edges()]

nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.4, edge_color=edge_colors, width=0.8)
nodes_drawn = nx.draw_networkx_nodes(
    G_sub, pos, ax=ax,
    node_color=node_colors, cmap=plt.cm.viridis,
    node_size=node_sizes, alpha=0.85, edgecolors="#333", linewidths=0.5
)

# Labels for high-degree nodes only
labels = {n: n for n, d in degrees.items() if d >= 6}
nx.draw_networkx_labels(G_sub, pos, labels, ax=ax, font_size=7, font_weight="bold")

ax.set_title(
    f"Physics Entity Network — Semantic Links (degree ≥ 4)\n"
    f"{len(G_sub.nodes())} entities, {len(G_sub.edges())} links",
    fontsize=16, fontweight="bold", pad=20
)

# Legend
legend_items = [
    mpatches.Patch(color="#4e79a7", label="has part(s) / part of"),
    mpatches.Patch(color="#e15759", label="measured phys. quantity"),
    mpatches.Patch(color="#f28e2c", label="different from"),
    mpatches.Patch(color="#76b7b2", label="same as"),
    mpatches.Patch(color="#ff9da7", label="opposite of"),
    mpatches.Patch(color="#b0b0b0", label="follows / followed by"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=10, framealpha=0.9)
ax.axis("off")
plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_network_sample.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig4_network_sample.png")


# ── 5. Class-to-Class Connectivity Heatmap ───────────────────────────────────
print("Generating fig5: Class connectivity heatmap...")
# For each link, determine source and target classes
class_cross = defaultdict(Counter)
for lnk in links:
    src_entity = entity_map.get(lnk["entity"], {})
    tgt_entity = entity_map.get(lnk["target"], {})
    src_classes = src_entity.get("classes", ["unknown"])
    tgt_classes = tgt_entity.get("classes", ["unknown"])
    for sc in src_classes:
        for tc in tgt_classes:
            class_cross[sc][tc] += 1

# Build matrix for top classes
top_classes = [c for c, _ in class_counts.most_common(12)]
matrix = np.zeros((len(top_classes), len(top_classes)))
for i, sc in enumerate(top_classes):
    for j, tc in enumerate(top_classes):
        matrix[i][j] = class_cross[sc][tc]

fig, ax = plt.subplots(figsize=(12, 10))
mask = matrix == 0
sns.heatmap(
    matrix, ax=ax,
    xticklabels=top_classes, yticklabels=top_classes,
    cmap="YlOrRd", annot=True, fmt=".0f",
    mask=mask, linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Number of Links"}
)
ax.set_title("Inter-Class Connectivity Heatmap", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Target Class", fontsize=12)
ax.set_ylabel("Source Class", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig5_class_connectivity.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig5_class_connectivity.png")


# ── 6. Top 25 Most Connected Entities ────────────────────────────────────────
print("Generating fig6: Top connected entities...")
fig, ax = plt.subplots(figsize=(14, 9))
top25 = degree.most_common(25)
labels_top = []
for qid, _ in top25:
    e = entity_map.get(qid, {})
    lbl = e.get("label", qid)
    cls = e.get("classes", [""])[0] if e.get("classes") else ""
    labels_top.append(f"{lbl}")

degrees_top = [d for _, d in top25]
classes_top = []
for qid, _ in top25:
    e = entity_map.get(qid, {})
    cls_list = e.get("classes", ["other"])
    classes_top.append(cls_list[0] if cls_list else "other")

# Color by class
unique_classes = list(set(classes_top))
class_palette = dict(zip(unique_classes, sns.color_palette("Set2", len(unique_classes))))
bar_colors = [class_palette[c] for c in classes_top]

bars = ax.barh(labels_top[::-1], degrees_top[::-1], color=bar_colors[::-1],
               edgecolor="white", linewidth=0.5)
ax.set_xlabel("Number of Inter-Entity Links", fontsize=13)
ax.set_title("Top 25 Most-Connected Physics Entities", fontsize=16, fontweight="bold", pad=15)

for bar, val in zip(bars, degrees_top[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9, color="#333")

# Legend
legend_patches = [mpatches.Patch(color=class_palette[c], label=c) for c in unique_classes]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9, title="Entity Class", framealpha=0.9)
ax.set_xlim(0, max(degrees_top) * 1.15)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig6_top_entities.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig6_top_entities.png")


# ── 7. Predicate Categories Donut ────────────────────────────────────────────
print("Generating fig7: Predicate categories...")
# Group predicates into semantic categories
pred_categories = {
    "Taxonomy": ["instance of", "subclass of"],
    "Composition": ["has part(s)", "part of", "facet of"],
    "Sequence": ["followed by", "follows"],
    "Identity": ["different from", "said to be the same as", "opposite of", "coextensive with"],
    "Measurement": ["measured physical quantity", "in defining formula", "defining formula",
                    "mass", "electric charge"],
    "Naming & History": ["named after", "discoverer or inventor", "time of discovery or invention"],
    "Causal / Usage": ["interaction", "has use", "used by", "uses", "has effect",
                       "has cause", "end cause", "based on", "influenced by"],
    "Study": ["studied by", "has characteristic"],
}

cat_counts = {}
for cat, preds in pred_categories.items():
    cat_counts[cat] = sum(pred_counts.get(p, 0) for p in preds)

fig, ax = plt.subplots(figsize=(10, 10))
cat_df = pd.DataFrame(
    [(cat, cnt) for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])],
    columns=["category", "count"]
)
colors7 = sns.color_palette("Set2", len(cat_df))
wedges, texts, autotexts = ax.pie(
    cat_df["count"],
    labels=[f"{row['category']}\n({row['count']:,})" for _, row in cat_df.iterrows()],
    colors=colors7,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.8,
    textprops={"fontsize": 11},
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight("bold")
ax.set_title(
    "Semantic Triple Categories\n(33,062 total triples grouped by predicate type)",
    fontsize=16, fontweight="bold", pad=20,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_predicate_categories.png", dpi=DPI, bbox_inches="tight")
plt.close()
print(f"  → fig7_predicate_categories.png")


# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All outputs saved to: {OUT_DIR}/")
print(f"  - summary_report.txt")
print(f"  - fig1_predicate_frequency.png")
print(f"  - fig2_entity_classes.png")
print(f"  - fig3_link_types.png")
print(f"  - fig4_network_sample.png")
print(f"  - fig5_class_connectivity.png")
print(f"  - fig6_top_entities.png")
print(f"  - fig7_predicate_categories.png")
print(f"{'='*60}")
