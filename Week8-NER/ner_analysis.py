#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Analysis using spaCy
Analyzes historical documents to extract named entities like people, organizations, locations, etc.
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
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text to analyze
        nlp (spacy.Language): Loaded spaCy model
        
    Returns:
        list: List of entities with their labels and descriptions
    """
    doc = nlp(text)
    entities = []
    
    # Only keep location-like entities: GPE (countries, cities, states) and LOC (non-GPE locations)
    location_labels = {"GPE", "LOC"}
    for ent in doc.ents:
        if ent.label_ not in location_labels:
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
    Save extracted entities to a single CSV file.
    
    Args:
        entities (list): List of entity dictionaries
        output_file (str): Output filename
    """
    if not entities:
        print("❌ No entities found to save.")
        return
    
    entities_df = pd.DataFrame(entities)
    entities_df.to_csv(output_file, index=False)
    print(f"💾 Saved {len(entities)} entities to: {output_file}")

def main():
    """Main function to run NER analysis."""
    parser = argparse.ArgumentParser(description='Named Entity Recognition Analysis using spaCy')
    parser.add_argument('input_file', help='Path to the text file to analyze')
    parser.add_argument('--model', default='en_core_web_sm', 
                       help='spaCy model to use (default: en_core_web_sm)')
    parser.add_argument('--output', default='ner_results', 
                       help='Base name for output files (default: ner_results)')
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
    print("🔍 Extracting location entities...")
    entities = extract_entities(text, nlp)
    
    if not entities:
        print("❌ No location entities found in the text.")
        return
    
    # Print summary
    print(f"\n📊 Found {len(entities)} location entities")
    
    # Save entities to CSV
    output_file = f"{args.output}.csv"
    save_entities_to_csv(entities, output_file)
    
    print(f"\n✅ Extraction complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
