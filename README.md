# Ontology Comparison

Compare REBEL models for knowledge graph triplet extraction from text.

## Models Compared

1. **Babelscape/rebel-large** - Original REBEL model trained on 200+ relation types
2. **konsman/rebel-quantum-mixed** - Fine-tuned version specialized for quantum physics domain

## Files

- `ontology_comparison.py` - Main comparison script
- `input_texts.txt` - Input sentences (one per line)
- `comparison_results.json` - Output with extracted triplets and differences

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run comparison
python ontology_comparison.py
```

## Input Format

Add sentences to `input_texts.txt`, one per line. Lines starting with `#` are treated as comments.

## Output Format

The `comparison_results.json` contains:

```json
{
  "timestamp": "...",
  "summary": {
    "total_texts": 15,
    "texts_with_differences": 3,
    "texts_identical": 12,
    "difference_indices": [2, 5, 8]
  },
  "differences": [
    {
      "text_index": 2,
      "text": "...",
      "comparison": {
        "identical": false,
        "only_in_first": [...],
        "only_in_second": [...],
        "common": [...]
      }
    }
  ],
  "all_results": [...]
}
```

### Quick Difference Check

To quickly see if models differ:
1. Check `summary.texts_with_differences` - if 0, models agree on everything
2. Check `summary.difference_indices` - list of text indices where models differ
3. Check `differences` array - detailed breakdown of what differs for each text

## Dependencies

```
transformers
torch
```
