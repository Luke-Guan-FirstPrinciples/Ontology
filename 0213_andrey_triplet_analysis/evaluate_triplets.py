"""
Triplet Quality Evaluator (OpenAI Responses API)
==================================================
Reads a CSV of (subject, predicate, object) triplets and asks an LLM to
score each one on semantic correctness, scientific accuracy, specificity,
and overall quality (1–5 scale).

Outputs:
  • Evaluated CSV with score columns appended
  • Console summary of score distributions

Usage:
    python3 evaluate_triplets.py                       # uses config.yaml
    python3 evaluate_triplets.py --input my.csv        # override input
    python3 evaluate_triplets.py --model gpt-4.1       # override model
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment ──────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ── JSON Schema for structured output ─────────────────────────────────────
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_score": {
            "type": "integer",
            "description": "1-5: Is the triplet semantically well-formed? (1=nonsensical, 5=perfectly clear)"
        },
        "scientific_score": {
            "type": "integer",
            "description": "1-5: Is the triplet scientifically/factually correct? (1=wrong, 5=verified fact)"
        },
        "specificity_score": {
            "type": "integer",
            "description": "1-5: How specific and informative is the triplet? (1=vague/generic, 5=precise and detailed)"
        },
        "overall_score": {
            "type": "integer",
            "description": "1-5: Overall quality of this triplet for a knowledge graph (1=useless, 5=excellent)"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation for the scores (1-2 sentences)"
        }
    },
    "required": ["semantic_score", "scientific_score", "specificity_score", "overall_score", "reasoning"],
    "additionalProperties": False
}

# ── Prompt template ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert evaluator of knowledge-graph triplets extracted from scientific papers.

Your task: evaluate whether a given (subject, predicate, object) triplet is semantically well-formed and scientifically correct.

Scoring rubric (each dimension is 1–5):

**Semantic Score** — Is the triplet grammatically and logically coherent?
  1 = Nonsensical or ungrammatical (e.g., "equation → is → follows")
  2 = Poorly formed; relationship is unclear
  3 = Understandable but awkward or ambiguous
  4 = Clear and well-formed
  5 = Perfectly clear, natural, unambiguous relationship

**Scientific Score** — Is the stated relationship factually/scientifically correct?
  1 = Factually wrong or contradicts established knowledge
  2 = Misleading or likely incorrect
  3 = Plausible but unverifiable or overly general
  4 = Likely correct based on domain knowledge
  5 = Verified, well-established scientific fact

**Specificity Score** — How informative is the triplet for a knowledge graph?
  1 = Extremely vague (e.g., "results → show → data")
  2 = Generic, adds little knowledge
  3 = Moderately informative
  4 = Specific and useful
  5 = Highly precise, captures a concrete, citable relationship

**Overall Score** — Holistic quality for inclusion in a scientific knowledge graph
  1 = Should be discarded
  2 = Low quality, needs significant revision
  3 = Acceptable but not ideal
  4 = Good quality, useful for a knowledge graph
  5 = Excellent, high-value triplet

Return your evaluation as a JSON object with keys: semantic_score, scientific_score, specificity_score, overall_score, reasoning.\
"""


def build_user_prompt(row: pd.Series, col_cfg: dict) -> str:
    """Build the user prompt for a single triplet."""
    subj = row[col_cfg["subject"]]
    pred = row[col_cfg["predicate"]]
    obj = row[col_cfg["object"]]

    prompt = f"Evaluate this triplet:\n\n"
    prompt += f"  Subject:   {subj}\n"
    prompt += f"  Predicate: {pred}\n"
    prompt += f"  Object:    {obj}\n"

    # Add optional context if available
    conf_col = col_cfg.get("confidence")
    if conf_col and conf_col in row.index and pd.notna(row.get(conf_col)):
        prompt += f"\n  [Extraction confidence: {row[conf_col]}]"

    sec_col = col_cfg.get("section")
    if sec_col and sec_col in row.index and pd.notna(row.get(sec_col)):
        prompt += f"\n  [Source section: {row[sec_col]}]"

    return prompt


REASONING_MODELS = {"gpt-5", "gpt-5.2", "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o4-mini"}


def _is_reasoning_model(model: str) -> bool:
    """Check if the model is a reasoning model that doesn't support temperature."""
    base = model.split("-202")[0]  # strip date suffix like -2025-04-14
    return base in REASONING_MODELS or model in REASONING_MODELS


def evaluate_single(client: OpenAI, model: str, user_prompt: str,
                    eval_cfg: dict) -> dict:
    """Call the OpenAI Responses API for a single triplet evaluation."""
    reasoning_effort = eval_cfg.get("reasoning_effort", "medium")
    temperature = eval_cfg.get("temperature", 0.0)
    store = eval_cfg.get("store", False)
    max_retries = eval_cfg.get("max_retries", 3)

    for attempt in range(1, max_retries + 1):
        try:
            # Build kwargs — reasoning models don't support temperature
            kwargs = dict(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "triplet_evaluation",
                        "strict": True,
                        "schema": EVALUATION_SCHEMA,
                    }
                },
                store=store,
            )

            if _is_reasoning_model(model):
                kwargs["reasoning"] = {
                    "effort": reasoning_effort,
                    "summary": "concise",
                }
            else:
                kwargs["temperature"] = temperature

            response = client.responses.create(**kwargs)

            # Extract the text output
            output_text = response.output_text
            result = json.loads(output_text)

            # Add usage info
            if response.usage:
                result["input_tokens"] = response.usage.input_tokens
                result["output_tokens"] = response.usage.output_tokens

            return result

        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}/{max_retries}] JSON parse error: {e}")
            if attempt == max_retries:
                return _error_result(f"JSON parse error: {e}")

        except Exception as e:
            print(f"    [attempt {attempt}/{max_retries}] API error: {e}")
            if attempt == max_retries:
                return _error_result(str(e))
            time.sleep(2 ** attempt)  # exponential backoff

    return _error_result("Max retries exceeded")


def _error_result(error_msg: str) -> dict:
    return {
        "semantic_score": None,
        "scientific_score": None,
        "specificity_score": None,
        "overall_score": None,
        "reasoning": f"ERROR: {error_msg}",
        "input_tokens": 0,
        "output_tokens": 0,
    }


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    eval_cfg = cfg["evaluation"]
    col_cfg = cfg["columns"]

    parser = argparse.ArgumentParser(description="Evaluate triplets via OpenAI")
    parser.add_argument("--input", type=str, default=eval_cfg["input_file"],
                        help="Input CSV file with triplets")
    parser.add_argument("--output", type=str, default=eval_cfg["output_file"],
                        help="Output CSV file with evaluation scores")
    parser.add_argument("--model", type=str, default=eval_cfg["model"],
                        help="OpenAI model to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of rows to evaluate (for testing)")
    args = parser.parse_args()

    input_path = Path(__file__).parent / args.input
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Load CSV ─────────────────────────────────────────────────────
    print(f"Loading triplets from {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} triplets")

    if args.limit:
        df = df.head(args.limit)
        print(f"  Limited to first {args.limit} rows")

    # Validate required columns
    for key in ["subject", "predicate", "object"]:
        col_name = col_cfg[key]
        if col_name not in df.columns:
            print(f"ERROR: Column '{col_name}' not found in CSV. Available: {list(df.columns)}")
            sys.exit(1)

    # ─── Initialize OpenAI client ─────────────────────────────────────
    client = OpenAI()
    model = args.model
    print(f"\n  Model: {model}")
    print(f"  Reasoning effort: {eval_cfg.get('reasoning_effort', 'medium')}")
    print(f"  Temperature: {eval_cfg.get('temperature', 0.0)}")
    print(f"  Store: {eval_cfg.get('store', False)}")

    # ─── Evaluate each triplet ────────────────────────────────────────
    print(f"\nEvaluating {len(df)} triplets ...\n")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    for idx, row in df.iterrows():
        subj = row[col_cfg["subject"]]
        pred = row[col_cfg["predicate"]]
        obj = row[col_cfg["object"]]

        print(f"  [{idx+1}/{len(df)}] ({subj} -> {pred} -> {obj})")

        user_prompt = build_user_prompt(row, col_cfg)
        result = evaluate_single(client, model, user_prompt, eval_cfg)

        if result["semantic_score"] is None:
            errors += 1
            print(f"         ERROR: {result['reasoning']}")
        else:
            print(f"         Scores: sem={result['semantic_score']} "
                  f"sci={result['scientific_score']} "
                  f"spec={result['specificity_score']} "
                  f"overall={result['overall_score']}")

        total_input_tokens += result.get("input_tokens", 0)
        total_output_tokens += result.get("output_tokens", 0)
        results.append(result)

    # ─── Merge results into DataFrame ─────────────────────────────────
    results_df = pd.DataFrame(results)
    output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved evaluated triplets to {output_path}")

    # ─── Summary statistics ───────────────────────────────────────────
    score_cols = ["semantic_score", "scientific_score", "specificity_score", "overall_score"]
    valid = output_df.dropna(subset=score_cols)

    print(f"\n{'═' * 60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Model:             {model}")
    print(f"  Triplets evaluated: {len(df)}")
    print(f"  Successful:        {len(valid)}")
    print(f"  Errors:            {errors}")
    print(f"  Total input tokens:  {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")

    if len(valid) > 0:
        print(f"\n  {'Score':<22} {'Mean':>6} {'Median':>7} {'Std':>6} {'Min':>4} {'Max':>4}")
        print(f"  {'─'*22} {'─'*6} {'─'*7} {'─'*6} {'─'*4} {'─'*4}")
        for col in score_cols:
            s = valid[col].astype(float)
            label = col.replace("_score", "").capitalize()
            print(f"  {label:<22} {s.mean():>6.2f} {s.median():>7.1f} {s.std():>6.2f} {s.min():>4.0f} {s.max():>4.0f}")

        # Distribution
        print(f"\n  Overall Score Distribution:")
        for score in range(1, 6):
            n = (valid["overall_score"] == score).sum()
            pct = n / len(valid) * 100
            bar = "█" * int(pct / 2)
            print(f"    {score}: {n:>4} ({pct:>5.1f}%) {bar}")

    print(f"\n{'═' * 60}")


if __name__ == "__main__":
    main()
