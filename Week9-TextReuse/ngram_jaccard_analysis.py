#!/usr/bin/env python3
"""
N-grams and Jaccard Similarity Analysis for LGBTQ Guidebook Locations

This script analyzes location titles from data_philly.csv to find similar
locations using n-grams and Jaccard similarity. It demonstrates how to
identify locations that might be the same despite variations in naming.

Author: Created for History 8510 at Clemson University
"""

import csv
import re
from typing import List, Set, Tuple, Dict

def clean_text(text: str) -> str:
    """Clean and normalize text for comparison."""
    if not text or text == 'NA':
        return ""
    
    text = str(text).lower().strip()
    
    # Remove punctuation that doesn't affect meaning
    text = re.sub(r'[.,!?;:]', '', text)
    
    # Normalize apostrophes/quotes
    text = text.replace("'", "").replace('"', '')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text

def create_ngrams(text: str, n: int = 2) -> Set[str]:
    """Create word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams

def create_shingles(text: str, k: int = 3) -> Set[str]:
    """Create character k-shingles from text."""
    if len(text) < k:
        return {text} if text else set()
    
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        shingles.add(shingle)
    
    return shingles

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def calculate_similarity(title1: str, title2: str) -> Dict[str, float]:
    """Calculate similarity between two location titles."""
    # Clean the titles
    clean1 = clean_text(title1)
    clean2 = clean_text(title2)
    
    # Create n-grams and shingles
    ngrams1 = create_ngrams(clean1, n=2)
    ngrams2 = create_ngrams(clean2, n=2)
    shingles1 = create_shingles(clean1, k=3)
    shingles2 = create_shingles(clean2, k=3)
    
    # Calculate similarities
    ngram_sim = jaccard_similarity(ngrams1, ngrams2)
    shingle_sim = jaccard_similarity(shingles1, shingles2)
    
    # Combined similarity (weighted average)
    combined_sim = 0.75 * shingle_sim + 0.25 * ngram_sim
    
    return {
        'ngram_similarity': ngram_sim,
        'shingle_similarity': shingle_sim,
        'combined_similarity': combined_sim
    }

def load_locations(filename: str) -> List[Dict]:
    """Load location data from CSV file."""
    locations = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['title'] and row['title'] != 'NA':
                locations.append({
                    'title': row['title'],
                    'address': row.get('streetaddress', ''),
                    'type': row.get('type', ''),
                    'city': row.get('city', ''),
                    'year': row.get('Year', '')
                })
    
    return locations

def find_similar_pairs(locations: List[Dict], threshold: float = 0.3) -> List[Dict]:
    """Find pairs of similar locations."""
    similar_pairs = []
    
    print(f"Comparing {len(locations)} locations...")
    
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i >= j:  # Avoid comparing with self and duplicate comparisons
                continue
            
            # Skip if titles are exactly the same (exact duplicates)
            if loc1['title'].lower().strip() == loc2['title'].lower().strip():
                continue
            
            similarities = calculate_similarity(loc1['title'], loc2['title'])
            
            if similarities['combined_similarity'] >= threshold:
                similar_pairs.append({
                    'title1': loc1['title'],
                    'title2': loc2['title'],
                    'address1': loc1['address'],
                    'address2': loc2['address'],
                    'ngram_sim': similarities['ngram_similarity'],
                    'shingle_sim': similarities['shingle_similarity'],
                    'combined_sim': similarities['combined_similarity']
                })
    
    # Sort by combined similarity
    similar_pairs.sort(key=lambda x: x['combined_sim'], reverse=True)
    
    return similar_pairs

def export_results_to_csv(similar_pairs: List[Dict], filename: str = "similarity_results.csv"):
    """Export similarity results to CSV for further analysis."""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['title1', 'title2', 'address1', 'address2', 
                     'ngram_similarity', 'shingle_similarity', 'combined_similarity']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for pair in similar_pairs:
            writer.writerow({
                'title1': pair['title1'],
                'title2': pair['title2'],
                'address1': pair['address1'],
                'address2': pair['address2'],
                'ngram_similarity': pair['ngram_sim'],
                'shingle_similarity': pair['shingle_sim'],
                'combined_similarity': pair['combined_sim']
            })
    
    print(f"Results exported to {filename}")

def main():
    """Main function to run the analysis."""
    print("N-GRAMS AND JACCARD SIMILARITY ANALYSIS")
    print("LGBTQ Guidebook Location Similarity Detection")
    print("=" * 60)
    
    try:
        # Load the data
        filename = '/Users/amandaregan/Library/CloudStorage/Dropbox/*Teaching/8510-CodeForClass/Week9-TextReuse/data_philly.csv'
        locations = load_locations(filename)
        
        print(f"Loaded {len(locations)} locations with titles")
        print()
        
        # Find similar pairs
        similar_pairs = find_similar_pairs(locations, threshold=0.2)
        
        print(f"Found {len(similar_pairs)} potentially similar location pairs")
        print()
        
        if similar_pairs:
            # Export to CSV
            export_results_to_csv(similar_pairs)
            
            # Show summary statistics
            combined_scores = [pair['combined_sim'] for pair in similar_pairs]
            ngram_scores = [pair['ngram_sim'] for pair in similar_pairs]
            shingle_scores = [pair['shingle_sim'] for pair in similar_pairs]
            
            print("SUMMARY STATISTICS:")
            print("-" * 20)
            print(f"Average combined similarity: {sum(combined_scores)/len(combined_scores):.3f}")
            print(f"Average n-gram similarity: {sum(ngram_scores)/len(ngram_scores):.3f}")
            print(f"Average shingle similarity: {sum(shingle_scores)/len(shingle_scores):.3f}")
            print()
            
            # Count by similarity ranges
            high_sim = sum(1 for s in combined_scores if s > 0.7)
            med_sim = sum(1 for s in combined_scores if 0.3 <= s <= 0.7)
            low_sim = sum(1 for s in combined_scores if s < 0.3)
            
            print("SIMILARITY DISTRIBUTION:")
            print(f"  High similarity (>0.7): {high_sim} pairs")
            print(f"  Medium similarity (0.3-0.7): {med_sim} pairs")
            print(f"  Low similarity (<0.3): {low_sim} pairs")
        else:
            print("No similar pairs found with the current threshold.")
            print("Try lowering the threshold or check the data.")
    
    except FileNotFoundError:
        print("Error: Could not find the data file.")
        print("Please make sure data_philly.csv is in the correct location.")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
