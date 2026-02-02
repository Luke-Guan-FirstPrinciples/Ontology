"""
Compare REBEL models for relation extraction:
1. Babelscape/rebel-large (original)
2. konsman/rebel-quantum-mixed (quantum physics fine-tuned)
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datetime import datetime
from pathlib import Path


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


def run_inference(model_data, text):
    """Run inference on a single text with a loaded model."""
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    gen_kwargs = {
        "max_length": 256,
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


def read_input_texts(filepath="input_texts.txt"):
    """Read input texts from file, one per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    texts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                texts.append(line)
    return texts


def main():
    # Read input texts from file
    input_file = "input_texts.txt"
    print(f"Reading input texts from {input_file}...")
    texts = read_input_texts(input_file)
    print(f"Found {len(texts)} input texts\n")
    
    # Models to compare
    model_names = [
        "Babelscape/rebel-large",
        "konsman/rebel-quantum-mixed",
    ]
    
    # Load models once
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    models = load_models(model_names)
    
    # Process each text
    all_results = []
    differences_found = []
    
    for idx, text in enumerate(texts, 1):
        print(f"\n{'='*60}")
        print(f"TEXT {idx}/{len(texts)}")
        print("="*60)
        print(f'"{text[:80]}{"..." if len(text) > 80 else ""}"')
        
        text_results = {}
        for model_name in model_names:
            try:
                triplets, raw_output = run_inference(models[model_name], text)
                text_results[model_name] = {
                    "triplets": triplets,
                    "raw_output": raw_output,
                    "error": None
                }
                print(f"\n{model_name}: {len(triplets)} triplet(s)")
                for t in triplets:
                    print(f"  - {t['head']} --[{t['type']}]--> {t['tail']}")
            except Exception as e:
                print(f"\nError with {model_name}: {e}")
                text_results[model_name] = {
                    "triplets": None,
                    "raw_output": None,
                    "error": str(e)
                }
        
        # Compare the two models
        comparison = compare_triplets(
            text_results[model_names[0]]["triplets"],
            text_results[model_names[1]]["triplets"]
        )
        
        if not comparison["identical"]:
            differences_found.append({
                "text_index": idx,
                "text": text,
                "comparison": comparison
            })
            print(f"\n⚠️  DIFFERENCE DETECTED!")
        else:
            print(f"\n✓ Models agree")
        
        all_results.append({
            "text_index": idx,
            "text": text,
            "model_results": text_results,
            "comparison": comparison
        })
    
    # Build summary
    summary = {
        "total_texts": len(texts),
        "texts_with_differences": len(differences_found),
        "texts_identical": len(texts) - len(differences_found),
        "difference_indices": [d["text_index"] for d in differences_found]
    }
    
    # Save results to file
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "models": model_names,
        "summary": summary,
        "differences": differences_found,
        "all_results": all_results
    }
    
    output_file = "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total texts processed: {summary['total_texts']}")
    print(f"Texts where models agree: {summary['texts_identical']}")
    print(f"Texts where models differ: {summary['texts_with_differences']}")
    
    if differences_found:
        print(f"\nDifferences found in texts: {summary['difference_indices']}")
        for diff in differences_found:
            print(f"\n--- Text {diff['text_index']} ---")
            print(f'"{diff["text"][:60]}..."')
            comp = diff["comparison"]
            if comp["only_in_first"]:
                print(f"  Only in {model_names[0]}:")
                for t in comp["only_in_first"]:
                    print(f"    + {t['head']} --[{t['type']}]--> {t['tail']}")
            if comp["only_in_second"]:
                print(f"  Only in {model_names[1]}:")
                for t in comp["only_in_second"]:
                    print(f"    + {t['head']} --[{t['type']}]--> {t['tail']}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
