"""
Zero-shot LLM-based ontology/relation extraction.

Configurable model support for OpenAI, Anthropic, or local models.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- CONFIGURATION ---
CONFIG = {
    # Provider: "openai", "anthropic", or "ollama"
    "provider": "openai",
    
    # Model name (examples for each provider)
    # OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
    # Anthropic: "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"
    # Ollama: "llama3", "mistral", "mixtral"
    "model": "gpt-4o",
    
    # Temperature (0 = deterministic, higher = more creative)
    "temperature": 0.0,
    
    # Input/output files
    "input_file": "input_texts.txt",
    "output_file": "llm_extraction_results.json",
}

# --- PROMPT ---
EXTRACTION_PROMPT = """You are an expert in knowledge graph construction and ontology extraction.

Your task is to extract semantic triplets (subject-predicate-object relationships) from the given text.

Rules:
1. Extract ALL meaningful relationships from the text
2. Each triplet should be: (head_entity, relation_type, tail_entity)
3. Use clear, concise relation types (e.g., "is_a", "created_by", "located_in", "proposed", "discovered")
4. Entities should be extracted as they appear or as commonly known
5. Include temporal relations when dates/years are mentioned
6. Include causal relations when cause-effect is described
7. Be comprehensive but avoid redundant or trivial triplets

Output format: Return ONLY a JSON array of triplets. Each triplet should be an object with "head", "type", and "tail" keys.

Example input: "Albert Einstein developed the theory of relativity in 1905."
Example output:
[
  {"head": "Albert Einstein", "type": "developed", "tail": "theory of relativity"},
  {"head": "theory of relativity", "type": "inception_year", "tail": "1905"}
]

Now extract triplets from this text:
"{text}"

Return ONLY the JSON array, no explanation or markdown formatting."""


def get_openai_client():
    """Get OpenAI client."""
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")


def get_anthropic_client():
    """Get Anthropic client."""
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")


def call_openai(client, model: str, prompt: str, temperature: float) -> str:
    """Call OpenAI API."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def call_anthropic(client, model: str, prompt: str, temperature: float) -> str:
    """Call Anthropic API."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_ollama(model: str, prompt: str, temperature: float) -> str:
    """Call Ollama local API."""
    try:
        import requests
    except ImportError:
        raise ImportError("requests package not installed. Run: pip install requests")
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_triplets_llm(
    text: str,
    provider: str,
    model: str,
    temperature: float,
    client=None
) -> tuple[list[dict], str, Optional[str]]:
    """
    Extract triplets using LLM.
    
    Returns: (triplets, raw_response, error)
    """
    prompt = EXTRACTION_PROMPT.format(text=text)
    
    try:
        if provider == "openai":
            if client is None:
                client = get_openai_client()
            raw_response = call_openai(client, model, prompt, temperature)
        elif provider == "anthropic":
            if client is None:
                client = get_anthropic_client()
            raw_response = call_anthropic(client, model, prompt, temperature)
        elif provider == "ollama":
            raw_response = call_ollama(model, prompt, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Parse JSON response
        # Handle potential markdown code blocks
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        triplets = json.loads(cleaned)
        
        # Validate structure
        validated = []
        for t in triplets:
            if isinstance(t, dict) and "head" in t and "type" in t and "tail" in t:
                validated.append({
                    "head": str(t["head"]),
                    "type": str(t["type"]),
                    "tail": str(t["tail"])
                })
        
        return validated, raw_response, None
        
    except Exception as e:
        return [], str(e) if 'raw_response' not in locals() else raw_response, str(e)


def read_input_texts(filepath: str) -> list[str]:
    """Read input texts from file, one per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    texts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                texts.append(line)
    return texts


def main():
    # Load configuration
    provider = CONFIG["provider"]
    model = CONFIG["model"]
    temperature = CONFIG["temperature"]
    input_file = CONFIG["input_file"]
    output_file = CONFIG["output_file"]
    
    print("=" * 60)
    print("LLM-BASED ONTOLOGY EXTRACTION")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Input file: {input_file}")
    print()
    
    # Show the prompt template
    print("=" * 60)
    print("PROMPT TEMPLATE")
    print("=" * 60)
    print(EXTRACTION_PROMPT.replace("{text}", "<INPUT_TEXT>"))
    print()
    
    # Read input texts
    print(f"Reading input texts from {input_file}...")
    texts = read_input_texts(input_file)
    print(f"Found {len(texts)} input texts\n")
    
    # Initialize client once (for efficiency)
    client = None
    if provider == "openai":
        client = get_openai_client()
        print("OpenAI client initialized")
    elif provider == "anthropic":
        client = get_anthropic_client()
        print("Anthropic client initialized")
    elif provider == "ollama":
        print("Using Ollama local API")
    
    # Process each text
    all_results = []
    total_triplets = 0
    errors = 0
    
    for idx, text in enumerate(texts, 1):
        print(f"\n{'='*60}")
        print(f"TEXT {idx}/{len(texts)}")
        print("=" * 60)
        print(f'"{text[:80]}{"..." if len(text) > 80 else ""}"')
        
        triplets, raw_response, error = extract_triplets_llm(
            text=text,
            provider=provider,
            model=model,
            temperature=temperature,
            client=client
        )
        
        if error:
            print(f"\n❌ Error: {error}")
            errors += 1
        else:
            print(f"\n✓ Extracted {len(triplets)} triplet(s)")
            for t in triplets:
                print(f"  - {t['head']} --[{t['type']}]--> {t['tail']}")
            total_triplets += len(triplets)
        
        all_results.append({
            "text_index": idx,
            "text": text,
            "triplets": triplets,
            "raw_response": raw_response,
            "error": error
        })
    
    # Build summary
    summary = {
        "total_texts": len(texts),
        "total_triplets": total_triplets,
        "average_triplets_per_text": round(total_triplets / len(texts), 2) if texts else 0,
        "errors": errors,
        "successful": len(texts) - errors
    }
    
    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "input_file": input_file,
        },
        "prompt_template": EXTRACTION_PROMPT,
        "summary": summary,
        "results": all_results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total texts processed: {summary['total_texts']}")
    print(f"Successful extractions: {summary['successful']}")
    print(f"Errors: {summary['errors']}")
    print(f"Total triplets extracted: {summary['total_triplets']}")
    print(f"Average triplets per text: {summary['average_triplets_per_text']}")
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
