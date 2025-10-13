#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Analysis using spaCy
Analyzes historical documents to extract people (PERSON entities).
"""

import spacy
import pandas as pd
import argparse
import sys
from pathlib import Path

def load_spacy_model(model_name="en_core_web_sm"):
    """
    Load spaCy model for NER analysis.
    
    Args:
        model_name (str): Name of the spaCy model to load
        
    Returns:
        spacy.Language: Loaded spaCy model
    """
    try:
        nlp = spacy.load(model_name)
        print(f"✓ Loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        print(f"❌ Error: spaCy model '{model_name}' not found.")
        print(f"Please install it with: python -m spacy download {model_name}")
        sys.exit(1)

def extract_entities(text, nlp):
    """
    Extract people entities from text using spaCy.
    
    Args:
        text (str): Input text to analyze
        nlp (spacy.Language): Loaded spaCy model
        
    Returns:
        list: List of people entities with their labels and descriptions
    """
    doc = nlp(text)
    entities = []
    
    # Only keep person entities: PERSON (people, including fictional)
    person_labels = {"PERSON"}
    for ent in doc.ents:
        if ent.label_ not in person_labels:
            continue
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'label_description': spacy.explain(ent.label_),
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': ent._.prob if hasattr(ent._, 'prob') else None
        })
    
    return entities

def save_entities_to_csv(entities, output_file):
    """
    Save extracted entities to a single CSV file in the results directory.
    
    Args:
        entities (list): List of entity dictionaries
        output_file (str): Output filename
    """
    if not entities:
        print("❌ No entities found to save.")
        return
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save to results directory
    output_path = results_dir / output_file
    entities_df = pd.DataFrame(entities)
    entities_df.to_csv(output_path, index=False)
    print(f"💾 Saved {len(entities)} entities to: {output_path}")

def main():
    """Main function to run NER analysis."""
    parser = argparse.ArgumentParser(description='Named Entity Recognition Analysis using spaCy - People Extraction')
    parser.add_argument('input_file', help='Path to the text file to analyze')
    parser.add_argument('--model', default='en_core_web_sm', 
                       help='spaCy model to use (default: en_core_web_sm)')
    parser.add_argument('--output', default='people_results', 
                       help='Base name for output files (default: people_results)')
    parser.add_argument('--max-chars', type=int, default=None,
                       help='Maximum characters to process (useful for large files)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    print(f"📖 Reading file: {args.input_file}")
    
    # Load spaCy model
    nlp = load_spacy_model(args.model)
    
    # Read text file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(input_path, 'r', encoding='latin-1') as f:
            text = f.read()
    
    # Limit text size if specified
    if args.max_chars and len(text) > args.max_chars:
        text = text[:args.max_chars]
        print(f"⚠️  Limited processing to first {args.max_chars} characters")
    
    print(f"📝 Processing {len(text):,} characters of text...")
    
    # Extract entities
    print("🔍 Extracting people entities...")
    entities = extract_entities(text, nlp)
    
    if not entities:
        print("❌ No people entities found in the text.")
        return
    
    # Print summary
    print(f"\n📊 Found {len(entities)} people entities")
    
    # Save entities to CSV
    output_file = f"{args.output}.csv"
    save_entities_to_csv(entities, output_file)
    
    print(f"\n✅ Extraction complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
