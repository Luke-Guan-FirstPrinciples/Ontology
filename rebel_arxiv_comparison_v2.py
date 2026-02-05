"""
Compare REBEL models for relation extraction on arXiv papers.
Models:
1. Babelscape/rebel-large (original)
2. konsman/rebel-quantum-mixed (quantum physics fine-tuned)

Data: 1000 randomly selected papers from PostgreSQL (arxiv_base.arxiv_from_kaggle)
"""

import os
import json
import yaml
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd

# Load environment variables
load_dotenv()


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_db_connection():
    """Create PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def fetch_arxiv_papers(limit=1000):
    """
    Fetch random papers from arxiv_base.arxiv_from_kaggle.
    Returns list of dicts with id, title, categories, abstract.
    """
    print(f"Connecting to PostgreSQL database...")
    conn = get_db_connection()
    
    query = """
        SELECT id, title, categories, abstract
        FROM arxiv_base.arxiv_from_kaggle
        WHERE abstract IS NOT NULL 
          AND abstract != ''
          AND title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT %s
    """
    
    print(f"Fetching {limit} random papers...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (limit,))
        papers = [dict(row) for row in cur.fetchall()]
    
    conn.close()
    print(f"Fetched {len(papers)} papers from database")
    return papers


def save_input_data(papers, filepath="arxiv_input_papers.json"):
    """Save fetched papers to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "source": "arxiv_base.arxiv_from_kaggle",
        "count": len(papers),
        "papers": papers
    }
    
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved input data to {filepath}")
    return filepath


def load_input_data(filepath="arxiv_input_papers.json"):
    """Load papers from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["papers"]


def extract_triplets(text):
    """Extract structured triplets from REBEL output."""
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    
    if subject != '' and relation != '' and object_ != '':
        triplets.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return triplets


def triplet_to_tuple(triplet):
    """Convert triplet dict to tuple for comparison."""
    return (triplet['head'], triplet['type'], triplet['tail'])


def compare_triplets(triplets1, triplets2):
    """Compare two lists of triplets and return differences."""
    if triplets1 is None or triplets2 is None:
        return {
            "identical": False,
            "only_in_first": [],
            "only_in_second": [],
            "common": [],
            "error": "One or both models failed"
        }
    
    set1 = set(triplet_to_tuple(t) for t in triplets1)
    set2 = set(triplet_to_tuple(t) for t in triplets2)
    
    common = set1 & set2
    only_in_first = set1 - set2
    only_in_second = set2 - set1
    
    return {
        "identical": set1 == set2,
        "only_in_first": [{"head": t[0], "type": t[1], "tail": t[2]} for t in only_in_first],
        "only_in_second": [{"head": t[0], "type": t[1], "tail": t[2]} for t in only_in_second],
        "common": [{"head": t[0], "type": t[1], "tail": t[2]} for t in common],
        "count_first": len(set1),
        "count_second": len(set2),
        "count_common": len(common),
    }


def load_models(model_names):
    """Load all models and tokenizers upfront."""
    models = {}
    for model_name in model_names:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        models[model_name] = {"tokenizer": tokenizer, "model": model}
    return models


def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """
    Split text into chunks that fit within max_length tokens.
    Returns list of text chunks with overlap for context continuity.
    """
    # Tokenize the full text to get token count
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_length:
        return [text], False  # No chunking needed
    
    chunks = []
    start = 0
    chunk_size = max_length - 10  # Leave room for special tokens
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        
        # Move start with overlap
        start = end - overlap
    
    return chunks, True


def run_inference_single(model_data, text, max_length=512, gen_kwargs=None):
    """Run inference on a single text chunk with a loaded model."""
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    if gen_kwargs is None:
        gen_kwargs = {
            "max_length": max_length,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 1,
        }
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **gen_kwargs
    )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    triplets = extract_triplets(raw_output)
    
    return triplets, raw_output


def run_inference(model_data, text, config):
    """
    Run inference on text, chunking if necessary.
    Returns combined triplets from all chunks.
    """
    tokenizer = model_data["tokenizer"]
    max_length = config["processing"]["max_length"]
    overlap = config["processing"]["chunk_overlap"]
    
    gen_kwargs = {
        "max_length": max_length,
        "length_penalty": config["generation"]["length_penalty"],
        "num_beams": config["generation"]["num_beams"],
        "num_return_sequences": config["generation"]["num_return_sequences"],
    }
    
    # Check if chunking is needed
    chunks, was_chunked = chunk_text(text, tokenizer, max_length, overlap)
    
    all_triplets = []
    all_raw_outputs = []
    
    for chunk in chunks:
        triplets, raw_output = run_inference_single(
            model_data, chunk, max_length, gen_kwargs
        )
        all_triplets.extend(triplets)
        all_raw_outputs.append(raw_output)
    
    # Deduplicate triplets (may have duplicates from overlapping chunks)
    seen = set()
    unique_triplets = []
    for t in all_triplets:
        key = (t['head'], t['type'], t['tail'])
        if key not in seen:
            seen.add(key)
            unique_triplets.append(t)
    
    combined_raw = " ||| ".join(all_raw_outputs) if was_chunked else all_raw_outputs[0]
    
    return unique_triplets, combined_raw, was_chunked, len(chunks)


def process_paper(paper, models, model_names, config):
    """Process a single paper through both models."""
    # Use abstract for relation extraction (it's more informative than title)
    text = paper.get("abstract", "")
    
    if not text:
        return None
    
    paper_results = {
        "paper_id": paper["id"],
        "title": paper["title"],
        "categories": paper["categories"],
        "abstract_length": len(text),
        "model_results": {}
    }
    
    for model_name in model_names:
        try:
            triplets, raw_output, was_chunked, num_chunks = run_inference(
                models[model_name], text, config
            )
            paper_results["model_results"][model_name] = {
                "triplets": triplets,
                "triplet_count": len(triplets),
                "raw_output": raw_output,
                "was_chunked": was_chunked,
                "num_chunks": num_chunks,
                "error": None
            }
        except Exception as e:
            paper_results["model_results"][model_name] = {
                "triplets": None,
                "triplet_count": 0,
                "raw_output": None,
                "was_chunked": False,
                "num_chunks": 0,
                "error": str(e)
            }
    
    # Compare the two models
    paper_results["comparison"] = compare_triplets(
        paper_results["model_results"][model_names[0]]["triplets"],
        paper_results["model_results"][model_names[1]]["triplets"]
    )
    
    return paper_results


def save_detailed_results_xlsx(all_results, model_names, output_path):
    """Save detailed results to Excel file."""
    rows = []
    
    for result in all_results:
        model1 = model_names[0]
        model2 = model_names[1]
        
        # Get model results
        m1_results = result["model_results"].get(model1, {})
        m2_results = result["model_results"].get(model2, {})
        
        # Format triplets as strings
        m1_triplets = m1_results.get("triplets") or []
        m2_triplets = m2_results.get("triplets") or []
        
        m1_triplets_str = "; ".join([
            f"({t['head']} -> {t['type']} -> {t['tail']})" 
            for t in m1_triplets
        ]) if m1_triplets else ""
        
        m2_triplets_str = "; ".join([
            f"({t['head']} -> {t['type']} -> {t['tail']})" 
            for t in m2_triplets
        ]) if m2_triplets else ""
        
        # Comparison details
        comparison = result.get("comparison", {})
        common_triplets = comparison.get("common", [])
        only_model1 = comparison.get("only_in_first", [])
        only_model2 = comparison.get("only_in_second", [])
        
        common_str = "; ".join([
            f"({t['head']} -> {t['type']} -> {t['tail']})" 
            for t in common_triplets
        ]) if common_triplets else ""
        
        only_m1_str = "; ".join([
            f"({t['head']} -> {t['type']} -> {t['tail']})" 
            for t in only_model1
        ]) if only_model1 else ""
        
        only_m2_str = "; ".join([
            f"({t['head']} -> {t['type']} -> {t['tail']})" 
            for t in only_model2
        ]) if only_model2 else ""
        
        row = {
            "paper_id": result.get("paper_id", ""),
            "title": result.get("title", ""),
            "categories": result.get("categories", ""),
            "abstract_length": result.get("abstract_length", 0),
            f"{model1}_triplet_count": m1_results.get("triplet_count", 0),
            f"{model1}_was_chunked": m1_results.get("was_chunked", False),
            f"{model1}_num_chunks": m1_results.get("num_chunks", 0),
            f"{model1}_triplets": m1_triplets_str,
            f"{model1}_error": m1_results.get("error", ""),
            f"{model2}_triplet_count": m2_results.get("triplet_count", 0),
            f"{model2}_was_chunked": m2_results.get("was_chunked", False),
            f"{model2}_num_chunks": m2_results.get("num_chunks", 0),
            f"{model2}_triplets": m2_triplets_str,
            f"{model2}_error": m2_results.get("error", ""),
            "identical": comparison.get("identical", False),
            "common_count": comparison.get("count_common", 0),
            "common_triplets": common_str,
            f"only_in_{model1}": only_m1_str,
            f"only_in_{model2}": only_m2_str,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Saved detailed results to {output_path}")
    return output_path


def generate_summary(all_results, model_names):
    """Generate summary statistics from all results."""
    total = len(all_results)
    identical = sum(1 for r in all_results if r["comparison"]["identical"])
    
    # Aggregate triplet counts
    total_triplets_model1 = sum(r["comparison"]["count_first"] for r in all_results)
    total_triplets_model2 = sum(r["comparison"]["count_second"] for r in all_results)
    total_common = sum(r["comparison"]["count_common"] for r in all_results)
    
    # Category breakdown
    category_stats = {}
    for r in all_results:
        cats = r.get("categories", "") or ""
        for cat in cats.split():
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "identical": 0}
            category_stats[cat]["count"] += 1
            if r["comparison"]["identical"]:
                category_stats[cat]["identical"] += 1
    
    # Sort categories by count
    top_categories = sorted(
        category_stats.items(), 
        key=lambda x: x[1]["count"], 
        reverse=True
    )[:20]
    
    return {
        "total_papers": total,
        "papers_identical": identical,
        "papers_with_differences": total - identical,
        "agreement_rate": round(identical / total * 100, 2) if total > 0 else 0,
        "total_triplets": {
            model_names[0]: total_triplets_model1,
            model_names[1]: total_triplets_model2,
            "common": total_common
        },
        "avg_triplets_per_paper": {
            model_names[0]: round(total_triplets_model1 / total, 2) if total > 0 else 0,
            model_names[1]: round(total_triplets_model2 / total, 2) if total > 0 else 0,
        },
        "top_categories": dict(top_categories)
    }


def main():
    # Load configuration from YAML file
    print("=" * 60)
    print("LOADING CONFIGURATION")
    print("=" * 60)
    config = load_config("config.yaml")
    print(f"Loaded config: max_length={config['processing']['max_length']}, "
          f"chunk_overlap={config['processing']['chunk_overlap']}")
    
    # Extract settings from config
    SAMPLE_SIZE = config["data"]["sample_size"]
    INPUT_FILE = config["data"]["input_file"]
    OUTPUT_JSON = config["data"]["output_json"]
    OUTPUT_XLSX = config["data"]["output_xlsx"]
    model_names = config["models"]
    
    # Step 1: Fetch data from database (or load from file if exists)
    print("\n" + "=" * 60)
    print("STEP 1: LOADING ARXIV DATA")
    print("=" * 60)
    
    input_path = Path(INPUT_FILE)
    if input_path.exists():
        print(f"Found existing input file: {INPUT_FILE}")
        user_input = input("Use existing file? (y/n): ").strip().lower()
        if user_input == 'y':
            papers = load_input_data(INPUT_FILE)
        else:
            papers = fetch_arxiv_papers(limit=SAMPLE_SIZE)
            save_input_data(papers, INPUT_FILE)
    else:
        papers = fetch_arxiv_papers(limit=SAMPLE_SIZE)
        save_input_data(papers, INPUT_FILE)
    
    print(f"Working with {len(papers)} papers")
    
    # Step 2: Load models
    print("\n" + "=" * 60)
    print("STEP 2: LOADING MODELS")
    print("=" * 60)
    models = load_models(model_names)
    
    # Step 3: Process papers
    print("\n" + "=" * 60)
    print("STEP 3: PROCESSING PAPERS")
    print("=" * 60)
    
    all_results = []
    differences = []
    chunked_count = 0
    
    for paper in tqdm(papers, desc="Processing papers"):
        result = process_paper(paper, models, model_names, config)
        if result:
            all_results.append(result)
            if not result["comparison"]["identical"]:
                differences.append({
                    "paper_id": result["paper_id"],
                    "title": result["title"],
                    "categories": result["categories"],
                    "comparison": result["comparison"]
                })
            # Track chunked papers
            for model_name in model_names:
                if result["model_results"][model_name].get("was_chunked"):
                    chunked_count += 1
                    break
    
    # Step 4: Generate summary
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING SUMMARY")
    print("=" * 60)
    
    summary = generate_summary(all_results, model_names)
    summary["papers_chunked"] = chunked_count
    
    # Step 5: Save results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": INPUT_FILE,
            "models": model_names,
            "total_papers_processed": len(all_results),
            "config": config
        },
        "summary": summary,
        "differences_overview": {
            "count": len(differences),
            "papers": differences
        },
        "detailed_results": all_results
    }
    
    # Save JSON output
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Save Excel output for detailed results
    print("\n" + "=" * 60)
    print("STEP 5: SAVING OUTPUTS")
    print("=" * 60)
    save_detailed_results_xlsx(all_results, model_names, OUTPUT_XLSX)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total papers processed: {summary['total_papers']}")
    print(f"Papers that required chunking: {summary['papers_chunked']}")
    print(f"Papers where models agree: {summary['papers_identical']}")
    print(f"Papers where models differ: {summary['papers_with_differences']}")
    print(f"Agreement rate: {summary['agreement_rate']}%")
    print(f"\nTotal triplets extracted:")
    print(f"  - {model_names[0]}: {summary['total_triplets'][model_names[0]]}")
    print(f"  - {model_names[1]}: {summary['total_triplets'][model_names[1]]}")
    print(f"  - Common: {summary['total_triplets']['common']}")
    print(f"\nAverage triplets per paper:")
    print(f"  - {model_names[0]}: {summary['avg_triplets_per_paper'][model_names[0]]}")
    print(f"  - {model_names[1]}: {summary['avg_triplets_per_paper'][model_names[1]]}")
    
    print(f"\n{'=' * 60}")
    print(f"Input data saved to: {INPUT_FILE}")
    print(f"JSON results saved to: {OUTPUT_JSON}")
    print(f"Excel results saved to: {OUTPUT_XLSX}")
    print("=" * 60)


if __name__ == "__main__":
    main()
