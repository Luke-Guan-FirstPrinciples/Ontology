"""
REBEL Model Comparison — Summary Analysis & Visualization
==========================================================
Reads rebel_comparison_results.xlsx and produces:
  • Console summary statistics
  • Predicate / Subject / Object frequency tables
  • Histograms & bar charts (saved as PNG + shown interactively)
  • Per-category breakdown of konsman/rebel-quantum-mixed triplet counts

Usage:
    python rebel_analysis.py
"""

import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────
INPUT_XLSX = "rebel_comparison_results.xlsx"
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_BABEL = "Babelscape/rebel-large"
MODEL_KONS = "konsman/rebel-quantum-mixed"

# Style
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {MODEL_BABEL: "#4C72B0", MODEL_KONS: "#DD8452"}
FIG_DPI = 150


# ── Helper: parse triplet strings ─────────────────────────────────────────
def parse_triplets(triplet_str):
    """
    Parse triplet strings in the form:
      (subject -> predicate -> object); (subject2 -> predicate2 -> object2)
    Returns list of dicts with keys: subject, predicate, object
    """
    if pd.isna(triplet_str) or not triplet_str.strip():
        return []

    triplets = []
    # Match each parenthesised triplet
    pattern = r"\(([^)]+)\)"
    for match in re.finditer(pattern, triplet_str):
        parts = match.group(1).split("->")
        if len(parts) == 3:
            triplets.append({
                "subject": parts[0].strip(),
                "predicate": parts[1].strip(),
                "object": parts[2].strip(),
            })
    return triplets


def explode_triplets(df, triplet_col, model_label):
    """Return a long-form DataFrame with one row per triplet."""
    rows = []
    for _, r in df.iterrows():
        triplets = parse_triplets(r[triplet_col])
        for t in triplets:
            rows.append({
                "paper_id": r["paper_id"],
                "categories": r["categories"],
                "model": model_label,
                **t,
            })
    return pd.DataFrame(rows)


def primary_category(cats_str):
    """Return the first (primary) arXiv category."""
    if pd.isna(cats_str):
        return "unknown"
    return cats_str.strip().split()[0]


def category_domain(cat):
    """Map an arXiv primary category to a coarse domain label."""
    prefixes = {
        "astro-ph": "Astrophysics",
        "cond-mat": "Condensed Matter",
        "cs": "Computer Science",
        "econ": "Economics",
        "eess": "Elec. Eng. & Sys. Sci.",
        "gr-qc": "General Relativity",
        "hep": "High Energy Physics",
        "math": "Mathematics",
        "math-ph": "Math Physics",
        "nlin": "Nonlinear Sciences",
        "nucl": "Nuclear Physics",
        "physics": "Physics (misc)",
        "q-bio": "Quantitative Biology",
        "q-fin": "Quantitative Finance",
        "quant-ph": "Quantum Physics",
        "stat": "Statistics",
    }
    for prefix, domain in prefixes.items():
        if cat.startswith(prefix):
            return domain
    return "Other"


# ── Pretty printing helpers ───────────────────────────────────────────────
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
    print(f"\n  {'Rank':<5} {label:<50} {'Count':>7} {'%':>7}")
    print(f"  {'─'*5} {'─'*50} {'─'*7} {'─'*7}")
    for i, (item, count) in enumerate(counter.most_common(top_n), 1):
        pct = count / total * 100 if total else 0
        display = (item[:47] + "...") if len(item) > 50 else item
        print(f"  {i:<5} {display:<50} {count:>7} {pct:>6.1f}%")
    print(f"  {'─'*5} {'─'*50} {'─'*7} {'─'*7}")
    print(f"  {'':5} {'TOTAL':<50} {total:>7}")
    print(f"  {'':5} {'Unique values':<50} {len(counter):>7}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # ─── Load data ─────────────────────────────────────────────────────
    print(f"Loading {INPUT_XLSX} ...")
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    n_papers = len(df)
    print(f"Loaded {n_papers} papers.\n")

    # Add derived columns
    df["primary_cat"] = df["categories"].apply(primary_category)
    df["domain"] = df["primary_cat"].apply(category_domain)

    # Explode triplets into long-form
    babel_trips = explode_triplets(df, f"{MODEL_BABEL}_triplets", MODEL_BABEL)
    kons_trips = explode_triplets(df, f"{MODEL_KONS}_triplets", MODEL_KONS)
    all_trips = pd.concat([babel_trips, kons_trips], ignore_index=True)

    # Add derived category columns to triplet frames
    for frame in (babel_trips, kons_trips, all_trips):
        frame["primary_cat"] = frame["categories"].apply(primary_category)
        frame["domain"] = frame["primary_cat"].apply(category_domain)

    # ===================================================================
    # 1. HIGH-LEVEL SUMMARY
    # ===================================================================
    section("1. HIGH-LEVEL SUMMARY")

    babel_count_col = f"{MODEL_BABEL}_triplet_count"
    kons_count_col = f"{MODEL_KONS}_triplet_count"

    stats = {
        "Metric": [
            "Total papers",
            "Papers with identical triplets",
            "Agreement rate (%)",
            f"Total triplets — {MODEL_BABEL}",
            f"Total triplets — {MODEL_KONS}",
            f"Avg triplets/paper — {MODEL_BABEL}",
            f"Avg triplets/paper — {MODEL_KONS}",
            f"Median triplets/paper — {MODEL_BABEL}",
            f"Median triplets/paper — {MODEL_KONS}",
            f"Max triplets/paper — {MODEL_BABEL}",
            f"Max triplets/paper — {MODEL_KONS}",
            "Papers where konsman > babel",
            "Papers where babel > konsman",
            "Papers where both = 0 triplets",
        ],
        "Value": [
            n_papers,
            int(df["identical"].sum()),
            round(df["identical"].mean() * 100, 2),
            int(df[babel_count_col].sum()),
            int(df[kons_count_col].sum()),
            round(df[babel_count_col].mean(), 3),
            round(df[kons_count_col].mean(), 3),
            int(df[babel_count_col].median()),
            int(df[kons_count_col].median()),
            int(df[babel_count_col].max()),
            int(df[kons_count_col].max()),
            int((df[kons_count_col] > df[babel_count_col]).sum()),
            int((df[babel_count_col] > df[kons_count_col]).sum()),
            int(((df[babel_count_col] == 0) & (df[kons_count_col] == 0)).sum()),
        ],
    }
    summary_df = pd.DataFrame(stats)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "summary_stats.csv", index=False)

    # ===================================================================
    # 2. PREDICATE ANALYSIS (FOCUS)
    # ===================================================================
    section("2. PREDICATE FREQUENCY ANALYSIS")

    for model_label, trips_df in [(MODEL_BABEL, babel_trips), (MODEL_KONS, kons_trips)]:
        subsection(f"Predicates — {model_label}")
        pred_counter = Counter(trips_df["predicate"].str.lower())
        print_counter_table(pred_counter, "Predicate", top_n=30)

    # Combined predicate comparison
    subsection("Predicate overlap between models")
    babel_preds = set(babel_trips["predicate"].str.lower().unique())
    kons_preds = set(kons_trips["predicate"].str.lower().unique())
    shared = babel_preds & kons_preds
    only_babel = babel_preds - kons_preds
    only_kons = kons_preds - babel_preds
    print(f"  Predicates unique to {MODEL_BABEL}: {len(only_babel)}")
    print(f"  Predicates unique to {MODEL_KONS}: {len(only_kons)}")
    print(f"  Shared predicates:                  {len(shared)}")
    if shared:
        print(f"  Shared: {sorted(shared)}")
    if only_kons:
        print(f"\n  Predicates exclusive to konsman ({len(only_kons)}):")
        for p in sorted(only_kons):
            print(f"    • {p}")
    if only_babel:
        print(f"\n  Predicates exclusive to Babelscape ({len(only_babel)}):")
        for p in sorted(only_babel):
            print(f"    • {p}")

    # ===================================================================
    # 3. SUBJECT & OBJECT ANALYSIS
    # ===================================================================
    section("3. SUBJECT & OBJECT FREQUENCY ANALYSIS")

    for model_label, trips_df in [(MODEL_BABEL, babel_trips), (MODEL_KONS, kons_trips)]:
        subsection(f"Top Subjects — {model_label}")
        subj_counter = Counter(trips_df["subject"].str.lower())
        print_counter_table(subj_counter, "Subject", top_n=20)

        subsection(f"Top Objects — {model_label}")
        obj_counter = Counter(trips_df["object"].str.lower())
        print_counter_table(obj_counter, "Object", top_n=20)

    # ===================================================================
    # 4. KONSMAN TRIPLET COUNT PER PAPER PER CATEGORY
    # ===================================================================
    section("4. KONSMAN TRIPLET COUNT — PER PAPER PER CATEGORY")

    cat_stats = (
        df.groupby("primary_cat")[kons_count_col]
        .agg(["count", "sum", "mean", "median", "std", "max"])
        .rename(columns={
            "count": "n_papers",
            "sum": "total_triplets",
            "mean": "avg_triplets",
            "median": "med_triplets",
            "std": "std_triplets",
            "max": "max_triplets",
        })
        .sort_values("total_triplets", ascending=False)
    )
    cat_stats["avg_triplets"] = cat_stats["avg_triplets"].round(2)
    cat_stats["std_triplets"] = cat_stats["std_triplets"].round(2)
    cat_stats["med_triplets"] = cat_stats["med_triplets"].astype(int)
    cat_stats["max_triplets"] = cat_stats["max_triplets"].astype(int)
    print(cat_stats.head(30).to_string())
    cat_stats.to_csv(OUTPUT_DIR / "konsman_triplets_per_category.csv")

    # Also by domain
    subsection("Aggregated by domain")
    domain_stats = (
        df.groupby("domain")[kons_count_col]
        .agg(["count", "sum", "mean", "median", "max"])
        .rename(columns={
            "count": "n_papers",
            "sum": "total_triplets",
            "mean": "avg_triplets",
            "median": "med_triplets",
            "max": "max_triplets",
        })
        .sort_values("total_triplets", ascending=False)
    )
    domain_stats["avg_triplets"] = domain_stats["avg_triplets"].round(2)
    print(domain_stats.to_string())
    domain_stats.to_csv(OUTPUT_DIR / "konsman_triplets_per_domain.csv")

    # ===================================================================
    # 5. VISUALISATIONS
    # ===================================================================
    section("5. GENERATING VISUALISATIONS")

    # ── Fig 1: Side-by-side triplet count distributions ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (model, col, color) in zip(axes, [
        (MODEL_BABEL, babel_count_col, PALETTE[MODEL_BABEL]),
        (MODEL_KONS, kons_count_col, PALETTE[MODEL_KONS]),
    ]):
        counts = df[col]
        max_val = int(counts.max())
        bins = np.arange(-0.5, max_val + 1.5, 1)
        ax.hist(counts, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Triplet count per paper")
        ax.set_ylabel("Number of papers")
        short = model.split("/")[-1]
        ax.set_title(f"{short}\n(μ={counts.mean():.2f}, med={int(counts.median())})")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("Triplet Count Distribution per Paper", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_triplet_count_dist.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig1_triplet_count_dist.png")

    # ── Fig 2: Top predicates (horizontal bar, both models) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    top_n_preds = 20

    for ax, (model_label, trips_df, color) in zip(axes, [
        (MODEL_BABEL, babel_trips, PALETTE[MODEL_BABEL]),
        (MODEL_KONS, kons_trips, PALETTE[MODEL_KONS]),
    ]):
        pred_counts = Counter(trips_df["predicate"].str.lower())
        top = pred_counts.most_common(top_n_preds)
        labels, vals = zip(*top) if top else ([], [])
        y_pos = np.arange(len(labels))

        ax.barh(y_pos, vals, color=color, edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        short = model_label.split("/")[-1]
        ax.set_title(f"Top {top_n_preds} Predicates — {short}", fontweight="bold")

    fig.suptitle("Predicate Frequency Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_predicate_frequency.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig2_predicate_frequency.png")

    # ── Fig 3: Konsman predicate diversity radar / treemap-style ──────
    # Predicate category heatmap: predicates × domain
    kons_pred_counts = Counter(kons_trips["predicate"].str.lower())
    top_kons_preds = [p for p, _ in kons_pred_counts.most_common(15)]

    if top_kons_preds and len(kons_trips) > 0:
        pivot_data = (
            kons_trips.assign(pred_lower=kons_trips["predicate"].str.lower())
            .query("pred_lower in @top_kons_preds")
            .groupby(["pred_lower", "domain"])
            .size()
            .reset_index(name="count")
            .pivot_table(index="pred_lower", columns="domain", values="count", fill_value=0)
        )
        # Reorder by total frequency
        pivot_data = pivot_data.loc[
            pivot_data.sum(axis=1).sort_values(ascending=False).index
        ]

        fig, ax = plt.subplots(figsize=(14, 7))
        pivot_data = pivot_data.astype(int)
        sns.heatmap(
            pivot_data, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Count"}
        )
        ax.set_xlabel("Domain")
        ax.set_ylabel("Predicate")
        ax.set_title(
            "konsman/rebel-quantum-mixed — Predicate × Domain Heatmap",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig3_predicate_domain_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig3_predicate_domain_heatmap.png")

    # ── Fig 4: Konsman triplet count by domain (box plot) ────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    domain_order = (
        df.groupby("domain")[kons_count_col]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=df, x="domain", y=kons_count_col,
        order=domain_order, palette="Set2", ax=ax, showfliers=True
    )
    ax.set_xlabel("")
    ax.set_ylabel("Triplet count per paper")
    ax.set_title(
        "konsman/rebel-quantum-mixed — Triplet Count Distribution by Domain",
        fontsize=13, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_konsman_domain_boxplot.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig4_konsman_domain_boxplot.png")

    # ── Fig 5: Top categories — Konsman avg triplets bar chart ───────
    top_cats = cat_stats.head(20)
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(top_cats))
    bars = ax.bar(
        x_pos, top_cats["avg_triplets"],
        color=PALETTE[MODEL_KONS], edgecolor="white", alpha=0.85
    )
    # Annotate paper counts
    for i, (idx, row) in enumerate(top_cats.iterrows()):
        ax.text(
            i, row["avg_triplets"] + 0.05,
            f"n={int(row['n_papers'])}",
            ha="center", va="bottom", fontsize=7, color="gray"
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_cats.index, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Avg triplets per paper")
    ax.set_title(
        "konsman/rebel-quantum-mixed — Avg Triplet Count by Primary Category (Top 20)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_konsman_category_avg.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig5_konsman_category_avg.png")

    # ── Fig 6: Scatter — Babel vs Konsman triplet counts per paper ───
    fig, ax = plt.subplots(figsize=(8, 8))
    jitter_x = df[babel_count_col] + np.random.uniform(-0.15, 0.15, n_papers)
    jitter_y = df[kons_count_col] + np.random.uniform(-0.15, 0.15, n_papers)
    ax.scatter(jitter_x, jitter_y, alpha=0.35, s=20, c="#555555", edgecolors="none")
    max_lim = max(df[babel_count_col].max(), df[kons_count_col].max()) + 1
    ax.plot([0, max_lim], [0, max_lim], "r--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"Triplet count — {MODEL_BABEL.split('/')[-1]}")
    ax.set_ylabel(f"Triplet count — {MODEL_KONS.split('/')[-1]}")
    ax.set_title("Per-Paper Triplet Count: Babelscape vs Konsman", fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, max_lim)
    ax.set_ylim(-0.5, max_lim)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_scatter_babel_vs_konsman.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig6_scatter_babel_vs_konsman.png")

    # ── Fig 7: Subject & Object word-cloud-style top-20 bars ─────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    panels = [
        (axes[0, 0], "Subjects", babel_trips, "subject", MODEL_BABEL, PALETTE[MODEL_BABEL]),
        (axes[0, 1], "Subjects", kons_trips, "subject", MODEL_KONS, PALETTE[MODEL_KONS]),
        (axes[1, 0], "Objects", babel_trips, "object", MODEL_BABEL, PALETTE[MODEL_BABEL]),
        (axes[1, 1], "Objects", kons_trips, "object", MODEL_KONS, PALETTE[MODEL_KONS]),
    ]

    for ax, entity_type, trips_df, col, model_label, color in panels:
        counter = Counter(trips_df[col].str.lower())
        top = counter.most_common(15)
        if top:
            labels, vals = zip(*top)
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, vals, color=color, edgecolor="white", alpha=0.85)
            ax.set_yticks(y_pos)
            wrapped = [textwrap.shorten(l, width=40, placeholder="...") for l in labels]
            ax.set_yticklabels(wrapped, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        short = model_label.split("/")[-1]
        ax.set_title(f"Top 15 {entity_type} — {short}", fontsize=11, fontweight="bold")

    fig.suptitle("Subject & Object Entity Frequencies", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_subject_object_bars.png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved fig7_subject_object_bars.png")

    # ── Fig 8: Predicate diversity per domain (stacked bar) ──────────
    if len(kons_trips) > 0:
        pred_diversity = (
            kons_trips.groupby("domain")["predicate"]
            .agg(["count", "nunique"])
            .rename(columns={"count": "total_triplets", "nunique": "unique_predicates"})
            .sort_values("total_triplets", ascending=False)
        )
        pred_diversity["avg_reuse"] = (
            pred_diversity["total_triplets"] / pred_diversity["unique_predicates"]
        ).round(2)

        fig, ax1 = plt.subplots(figsize=(14, 6))
        x_pos = np.arange(len(pred_diversity))
        width = 0.35

        bars1 = ax1.bar(
            x_pos - width / 2, pred_diversity["total_triplets"],
            width, color=PALETTE[MODEL_KONS], alpha=0.8, label="Total triplets"
        )
        bars2 = ax1.bar(
            x_pos + width / 2, pred_diversity["unique_predicates"],
            width, color="#55A868", alpha=0.8, label="Unique predicates"
        )

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(pred_diversity.index, rotation=40, ha="right", fontsize=9)
        ax1.set_ylabel("Count")
        ax1.set_title(
            "konsman/rebel-quantum-mixed — Predicate Volume & Diversity by Domain",
            fontsize=13, fontweight="bold"
        )
        ax1.legend(loc="upper right")

        # Overlay avg reuse as line
        ax2 = ax1.twinx()
        ax2.plot(
            x_pos, pred_diversity["avg_reuse"],
            "D-", color="#C44E52", markersize=6, linewidth=1.5, label="Avg reuse factor"
        )
        ax2.set_ylabel("Avg reuse (triplets / unique predicates)", color="#C44E52")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax2.legend(loc="upper center")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "fig8_predicate_diversity.png", dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved fig8_predicate_diversity.png")

    # ===================================================================
    # 6. SAVE FULL PREDICATE TABLES
    # ===================================================================
    section("6. SAVING DETAILED CSVs")

    for model_label, trips_df in [(MODEL_BABEL, babel_trips), (MODEL_KONS, kons_trips)]:
        short = model_label.split("/")[-1]
        # Predicate freq
        pred_freq = (
            pd.DataFrame(
                Counter(trips_df["predicate"].str.lower()).most_common(),
                columns=["predicate", "count"]
            )
        )
        pred_freq["pct"] = (pred_freq["count"] / pred_freq["count"].sum() * 100).round(2)
        pred_freq.to_csv(OUTPUT_DIR / f"{short}_predicate_freq.csv", index=False)

        # Subject freq
        subj_freq = (
            pd.DataFrame(
                Counter(trips_df["subject"].str.lower()).most_common(),
                columns=["subject", "count"]
            )
        )
        subj_freq.to_csv(OUTPUT_DIR / f"{short}_subject_freq.csv", index=False)

        # Object freq
        obj_freq = (
            pd.DataFrame(
                Counter(trips_df["object"].str.lower()).most_common(),
                columns=["object", "count"]
            )
        )
        obj_freq.to_csv(OUTPUT_DIR / f"{short}_object_freq.csv", index=False)

    print(f"  Saved predicate/subject/object frequency CSVs to {OUTPUT_DIR}/")

    # ===================================================================
    # DONE
    # ===================================================================
    section("ANALYSIS COMPLETE")
    print(f"\n  All outputs saved to:  {OUTPUT_DIR.resolve()}")
    print(f"  Figures: 8 PNG files")
    print(f"  Tables:  summary_stats.csv, konsman_triplets_per_category.csv,")
    print(f"           konsman_triplets_per_domain.csv, *_predicate_freq.csv,")
    print(f"           *_subject_freq.csv, *_object_freq.csv\n")


if __name__ == "__main__":
    main()
