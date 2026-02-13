"""
Graph RAG Triplet Analysis & Visualization
============================================
Reads from graph_rag.graph_triplets_2025_09_09 (PostgreSQL) and produces:
  • Console summary statistics
  • Predicate frequency tables & charts
  • Entity (subject + object) frequency tables & charts
  • Distribution histograms & bar charts (saved as PNG + shown interactively)

Usage:
    python3 triplet_analysis.py
"""

import os
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ── Load environment ──────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Configuration ─────────────────────────────────────────────────────────
TABLE = "graph_rag.graph_triplets_2025_09_09"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.05)
COLOR_PRIMARY = "#4C72B0"
COLOR_SECONDARY = "#DD8452"
COLOR_ACCENT = "#55A868"
COLOR_HIGHLIGHT = "#C44E52"
FIG_DPI = 150


# ── Database ──────────────────────────────────────────────────────────────
def get_db_connection():
    """Create PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def get_sqlalchemy_engine():
    """Create SQLAlchemy engine for pandas read_sql."""
    user = quote_plus(os.getenv("DB_USER"))
    password = quote_plus(os.getenv("DB_PASSWORD"))
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url)


def load_triplets() -> pd.DataFrame:
    """Load all triplets from the database into a DataFrame."""
    engine = get_sqlalchemy_engine()
    query = f"SELECT * FROM {TABLE}"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


# ── Pretty printing helpers ──────────────────────────────────────────────
def section(title, width=72):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def subsection(title, width=72):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_counter_table(counter, label, top_n=25):
    """Print a frequency table from a Counter."""
    total = sum(counter.values())
    print(f"\n  {'Rank':<5} {label:<55} {'Count':>8} {'%':>7}")
    print(f"  {'─'*5} {'─'*55} {'─'*8} {'─'*7}")
    for i, (item, count) in enumerate(counter.most_common(top_n), 1):
        pct = count / total * 100 if total else 0
        display = (item[:52] + "...") if len(str(item)) > 55 else str(item)
        print(f"  {i:<5} {display:<55} {count:>8} {pct:>6.1f}%")
    print(f"  {'─'*5} {'─'*55} {'─'*8} {'─'*7}")
    print(f"  {'':5} {'TOTAL':<55} {total:>8}")
    print(f"  {'':5} {'Unique values':<55} {len(counter):>8}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    # ─── Load data ────────────────────────────────────────────────────
    section("LOADING DATA")
    print(f"  Querying {TABLE} ...")
    df = load_triplets()
    n_triplets = len(df)
    print(f"  Loaded {n_triplets:,} triplets.")

    # Use canonical columns for analysis (lowercased / normalized)
    # Fall back to raw columns if canonical ones are empty
    has_canonical = df["canonical_predicate"].notna().sum() > 0
    pred_col = "canonical_predicate" if has_canonical else "predicate"
    subj_col = "canonical_subject" if has_canonical else "subject"
    obj_col = "canonical_object" if has_canonical else "object"
    print(f"  Using columns: {pred_col}, {subj_col}, {obj_col}")

    # ===================================================================
    # 1. HIGH-LEVEL SUMMARY
    # ===================================================================
    section("1. HIGH-LEVEL SUMMARY")

    n_papers = df["paper_id"].nunique()
    n_unique_predicates = df[pred_col].nunique()
    n_unique_subjects = df[subj_col].nunique()
    n_unique_objects = df[obj_col].nunique()

    # Combined unique entities (subjects + objects)
    all_entities = set(df[subj_col].dropna().unique()) | set(df[obj_col].dropna().unique())
    n_unique_entities = len(all_entities)

    # Triplets per paper
    trips_per_paper = df.groupby("paper_id").size()

    # Confidence stats
    has_confidence = df["confidence"].notna().sum() > 0

    stats = {
        "Metric": [
            "Total triplets",
            "Unique papers",
            "Unique predicates",
            "Unique subjects",
            "Unique objects",
            "Unique entities (subjects + objects)",
            "Avg triplets per paper",
            "Median triplets per paper",
            "Max triplets per paper",
            "Min triplets per paper",
            "Std dev triplets per paper",
        ],
        "Value": [
            f"{n_triplets:,}",
            f"{n_papers:,}",
            f"{n_unique_predicates:,}",
            f"{n_unique_subjects:,}",
            f"{n_unique_objects:,}",
            f"{n_unique_entities:,}",
            f"{trips_per_paper.mean():.2f}",
            f"{trips_per_paper.median():.0f}",
            f"{trips_per_paper.max():,}",
            f"{trips_per_paper.min():,}",
            f"{trips_per_paper.std():.2f}",
        ],
    }

    if has_confidence:
        stats["Metric"].extend([
            "Avg confidence",
            "Median confidence",
            "Min confidence",
        ])
        stats["Value"].extend([
            f"{df['confidence'].mean():.4f}",
            f"{df['confidence'].median():.4f}",
            f"{df['confidence'].min():.4f}",
        ])

    summary_df = pd.DataFrame(stats)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)

    # ===================================================================
    # 2. PREDICATE FREQUENCY ANALYSIS
    # ===================================================================
    section("2. PREDICATE FREQUENCY ANALYSIS")

    pred_counter = Counter(df[pred_col].dropna().str.strip())
    print_counter_table(pred_counter, "Predicate", top_n=30)

    # Save full predicate frequency
    pred_freq = pd.DataFrame(
        pred_counter.most_common(), columns=["predicate", "count"]
    )
    pred_freq["pct"] = (pred_freq["count"] / pred_freq["count"].sum() * 100).round(3)
    pred_freq.to_csv(OUTPUT_DIR / "predicate_frequency.csv", index=False)

    # ===================================================================
    # 3. SUBJECT FREQUENCY ANALYSIS
    # ===================================================================
    section("3. SUBJECT FREQUENCY ANALYSIS")

    subj_counter = Counter(df[subj_col].dropna().str.strip())
    print_counter_table(subj_counter, "Subject", top_n=25)

    subj_freq = pd.DataFrame(
        subj_counter.most_common(), columns=["subject", "count"]
    )
    subj_freq.to_csv(OUTPUT_DIR / "subject_frequency.csv", index=False)

    # ===================================================================
    # 4. OBJECT FREQUENCY ANALYSIS
    # ===================================================================
    section("4. OBJECT FREQUENCY ANALYSIS")

    obj_counter = Counter(df[obj_col].dropna().str.strip())
    print_counter_table(obj_counter, "Object", top_n=25)

    obj_freq = pd.DataFrame(
        obj_counter.most_common(), columns=["object", "count"]
    )
    obj_freq.to_csv(OUTPUT_DIR / "object_frequency.csv", index=False)

    # ===================================================================
    # 5. COMBINED ENTITY ANALYSIS (subjects + objects)
    # ===================================================================
    section("5. COMBINED ENTITY FREQUENCY (Subjects + Objects)")

    entity_counter = Counter()
    entity_counter.update(df[subj_col].dropna().str.strip())
    entity_counter.update(df[obj_col].dropna().str.strip())
    print_counter_table(entity_counter, "Entity", top_n=30)

    entity_freq = pd.DataFrame(
        entity_counter.most_common(), columns=["entity", "count"]
    )
    entity_freq.to_csv(OUTPUT_DIR / "entity_frequency.csv", index=False)

    # ===================================================================
    # 6. SECTION BREAKDOWN
    # ===================================================================
    if "section" in df.columns and df["section"].notna().sum() > 0:
        section("6. TRIPLETS BY SECTION")
        section_counter = Counter(df["section"].dropna().str.strip())
        print_counter_table(section_counter, "Section", top_n=20)

    # ===================================================================
    # 7. VISUALISATIONS
    # ===================================================================
    section("7. GENERATING VISUALISATIONS")

    # ── Fig 1: Top 25 Predicates (horizontal bar) ─────────────────────
    top_n_preds = 25
    top_preds = pred_counter.most_common(top_n_preds)
    if top_preds:
        labels, vals = zip(*top_preds)
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(y_pos, vals, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        wrapped = [textwrap.shorten(str(l), width=45, placeholder="...") for l in labels]
        ax.set_yticklabels(wrapped, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(
            f"Top {top_n_preds} Predicates — graph_triplets_2025_09_09",
            fontsize=13, fontweight="bold",
        )
        # Add count annotations
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.005, i, f"{v:,}", va="center", fontsize=8, color="gray")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig1_predicate_frequency.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig1_predicate_frequency.png")
        plt.close(fig)

    # ── Fig 2: Top 25 Subjects (horizontal bar) ──────────────────────
    top_n_subj = 25
    top_subjs = subj_counter.most_common(top_n_subj)
    if top_subjs:
        labels, vals = zip(*top_subjs)
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(y_pos, vals, color=COLOR_SECONDARY, edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        wrapped = [textwrap.shorten(str(l), width=45, placeholder="...") for l in labels]
        ax.set_yticklabels(wrapped, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(
            f"Top {top_n_subj} Subjects — graph_triplets_2025_09_09",
            fontsize=13, fontweight="bold",
        )
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.005, i, f"{v:,}", va="center", fontsize=8, color="gray")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig2_subject_frequency.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig2_subject_frequency.png")
        plt.close(fig)

    # ── Fig 3: Top 25 Objects (horizontal bar) ────────────────────────
    top_n_obj = 25
    top_objs = obj_counter.most_common(top_n_obj)
    if top_objs:
        labels, vals = zip(*top_objs)
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(y_pos, vals, color=COLOR_ACCENT, edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        wrapped = [textwrap.shorten(str(l), width=45, placeholder="...") for l in labels]
        ax.set_yticklabels(wrapped, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(
            f"Top {top_n_obj} Objects — graph_triplets_2025_09_09",
            fontsize=13, fontweight="bold",
        )
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.005, i, f"{v:,}", va="center", fontsize=8, color="gray")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig3_object_frequency.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig3_object_frequency.png")
        plt.close(fig)

    # ── Fig 4: Combined Entity Frequency — Top 30 ────────────────────
    top_n_ent = 30
    top_ents = entity_counter.most_common(top_n_ent)
    if top_ents:
        labels, vals = zip(*top_ents)
        y_pos = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(13, 10))
        # Color bars by whether entity appears more as subject or object
        colors = []
        for lbl in labels:
            s_count = subj_counter.get(lbl, 0)
            o_count = obj_counter.get(lbl, 0)
            if s_count > o_count:
                colors.append(COLOR_SECONDARY)  # subject-dominant
            elif o_count > s_count:
                colors.append(COLOR_ACCENT)      # object-dominant
            else:
                colors.append(COLOR_PRIMARY)      # balanced

        ax.barh(y_pos, vals, color=colors, edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        wrapped = [textwrap.shorten(str(l), width=45, placeholder="...") for l in labels]
        ax.set_yticklabels(wrapped, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency (as subject + as object)")
        ax.set_title(
            f"Top {top_n_ent} Entities (Subjects + Objects)",
            fontsize=13, fontweight="bold",
        )
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.005, i, f"{v:,}", va="center", fontsize=8, color="gray")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOR_SECONDARY, alpha=0.85, label="Subject-dominant"),
            Patch(facecolor=COLOR_ACCENT, alpha=0.85, label="Object-dominant"),
            Patch(facecolor=COLOR_PRIMARY, alpha=0.85, label="Balanced"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig4_entity_frequency.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig4_entity_frequency.png")
        plt.close(fig)

    # ── Fig 5: Triplets per Paper Distribution ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    max_val = int(np.percentile(trips_per_paper, 99))  # clip at 99th percentile for viz
    clipped = trips_per_paper.clip(upper=max_val)
    bins = np.arange(-0.5, max_val + 1.5, 1) if max_val <= 100 else 50
    ax.hist(clipped, bins=bins, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Triplets per paper")
    ax.set_ylabel("Number of papers")
    ax.set_title(
        f"Triplets per Paper Distribution\n"
        f"(μ={trips_per_paper.mean():.1f}, median={trips_per_paper.median():.0f}, "
        f"max={trips_per_paper.max():,}, clipped at p99={max_val})",
        fontsize=13, fontweight="bold",
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_triplets_per_paper_dist.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig5_triplets_per_paper_dist.png")
    plt.close(fig)

    # ── Fig 6: Confidence Distribution ────────────────────────────────
    if has_confidence:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            df["confidence"].dropna(), bins=50,
            color=COLOR_HIGHLIGHT, edgecolor="white", alpha=0.85,
        )
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Number of triplets")
        ax.set_title(
            f"Confidence Score Distribution\n"
            f"(μ={df['confidence'].mean():.3f}, median={df['confidence'].median():.3f})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig6_confidence_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig6_confidence_distribution.png")
        plt.close(fig)

    # ── Fig 7: Predicate Frequency — Log Scale (all predicates) ──────
    pred_vals = sorted(pred_counter.values(), reverse=True)
    if pred_vals:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(1, len(pred_vals) + 1), pred_vals, color=COLOR_PRIMARY, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Predicate Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(
            f"Predicate Frequency Distribution (Zipf Plot)\n"
            f"{n_unique_predicates:,} unique predicates",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig7_predicate_zipf.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig7_predicate_zipf.png")
        plt.close(fig)

    # ── Fig 8: Entity Frequency — Log Scale (all entities) ───────────
    entity_vals = sorted(entity_counter.values(), reverse=True)
    if entity_vals:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(1, len(entity_vals) + 1), entity_vals, color=COLOR_SECONDARY, linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Entity Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(
            f"Entity Frequency Distribution (Zipf Plot)\n"
            f"{n_unique_entities:,} unique entities (subjects + objects)",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig8_entity_zipf.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig8_entity_zipf.png")
        plt.close(fig)

    # ── Fig 9: Section Distribution (pie / bar) ──────────────────────
    if "section" in df.columns and df["section"].notna().sum() > 0:
        sec_counts = Counter(df["section"].dropna().str.strip())
        top_sections = sec_counts.most_common(15)
        if top_sections:
            labels, vals = zip(*top_sections)
            y_pos = np.arange(len(labels))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(y_pos, vals, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Number of triplets")
            ax.set_title(
                "Triplets by Section",
                fontsize=13, fontweight="bold",
            )
            for i, v in enumerate(vals):
                ax.text(v + max(vals) * 0.005, i, f"{v:,}", va="center", fontsize=8, color="gray")

            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "fig9_section_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
            print(f"  Saved fig9_section_distribution.png")
            plt.close(fig)

    # ── Fig 10: Predicate × Section Heatmap ──────────────────────────
    if "section" in df.columns and df["section"].notna().sum() > 0:
        top_preds_for_heatmap = [p for p, _ in pred_counter.most_common(15)]
        top_secs_for_heatmap = [s for s, _ in Counter(df["section"].dropna().str.strip()).most_common(10)]

        if top_preds_for_heatmap and top_secs_for_heatmap:
            pivot_data = (
                df.assign(pred_clean=df[pred_col].str.strip())
                .query("pred_clean in @top_preds_for_heatmap")
                .assign(sec_clean=df["section"].str.strip())
                .query("sec_clean in @top_secs_for_heatmap")
                .groupby(["pred_clean", "sec_clean"])
                .size()
                .reset_index(name="count")
                .pivot_table(index="pred_clean", columns="sec_clean", values="count", fill_value=0)
            )
            # Reorder by total frequency
            pivot_data = pivot_data.loc[
                pivot_data.sum(axis=1).sort_values(ascending=False).index
            ]
            pivot_data = pivot_data.astype(int)

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(
                pivot_data, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Count"},
            )
            ax.set_xlabel("Section")
            ax.set_ylabel("Predicate")
            ax.set_title(
                "Top Predicates × Section Heatmap",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "fig10_predicate_section_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
            print(f"  Saved fig10_predicate_section_heatmap.png")
            plt.close(fig)

    # ── Fig 11: Predicate Diversity per Paper ────────────────────────
    pred_diversity = df.groupby("paper_id")[pred_col].agg(
        total_triplets="count", unique_predicates="nunique"
    ).sort_values("total_triplets", ascending=False)
    pred_diversity["avg_reuse"] = (
        pred_diversity["total_triplets"] / pred_diversity["unique_predicates"]
    ).round(2)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(
        pred_diversity["total_triplets"],
        pred_diversity["unique_predicates"],
        alpha=0.25, s=15, c=COLOR_PRIMARY, edgecolors="none",
    )
    ax1.set_xlabel("Total triplets in paper")
    ax1.set_ylabel("Unique predicates in paper")
    ax1.set_title(
        "Predicate Diversity: Unique Predicates vs Total Triplets per Paper",
        fontsize=13, fontweight="bold",
    )
    # Add diagonal reference
    max_lim = max(pred_diversity["total_triplets"].max(), pred_diversity["unique_predicates"].max())
    ax1.plot([0, max_lim], [0, max_lim], "r--", alpha=0.4, label="y = x (max diversity)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig11_predicate_diversity_scatter.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig11_predicate_diversity_scatter.png")
    plt.close(fig)

    # ===================================================================
    # DONE
    # ===================================================================
    section("ANALYSIS COMPLETE")
    n_figs = len(list(OUTPUT_DIR.glob("fig*.png")))
    n_csvs = len(list(OUTPUT_DIR.glob("*.csv")))
    print(f"\n  All outputs saved to:  {OUTPUT_DIR.resolve()}")
    print(f"  Figures: {n_figs} PNG files")
    print(f"  Tables:  {n_csvs} CSV files")
    print(f"    - summary_stats.csv")
    print(f"    - predicate_frequency.csv")
    print(f"    - subject_frequency.csv")
    print(f"    - object_frequency.csv")
    print(f"    - entity_frequency.csv\n")


if __name__ == "__main__":
    main()
