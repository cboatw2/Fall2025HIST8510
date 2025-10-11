#!/usr/bin/env python3
"""
Demonstration Script for Word Vector Alignment

This script demonstrates how to use the word vector alignment tools
for analyzing semantic changes in historical texts.

Run this after training the models to see example results.
"""

import os
import pandas as pd
from pathlib import Path

def demonstrate_results():
    """Demonstrate the results of word vector alignment analysis."""
    
    print("Word Vector Alignment Demonstration")
    print("=" * 50)
    
    # Check if results exist
    results_dir = Path("alignment_results")
    if not results_dir.exists():
        print("❌ No alignment results found.")
        print("Please run the following commands first:")
        print("  1. python train_word2vec_models.py")
        print("  2. python align_word_vectors.py")
        return
    
    # Load and display results
    try:
        # Load semantic changes
        changes_file = results_dir / "semantic_changes.csv"
        if changes_file.exists():
            df = pd.read_csv(changes_file)
            
            print(f"✅ Analysis completed for {len(df)} words")
            print()
            
            # Show top changes
            print("🔍 Top 10 Words with Largest Semantic Changes:")
            print("-" * 50)
            top_changes = df.head(10)
            for _, row in top_changes.iterrows():
                print(f"  {row['word']:<15} {row['semantic_change']:.4f}")
            
            print()
            print("🔍 Top 10 Words with Smallest Semantic Changes:")
            print("-" * 50)
            bottom_changes = df.tail(10)
            for _, row in bottom_changes.iterrows():
                print(f"  {row['word']:<15} {row['semantic_change']:.4f}")
            
            print()
            print("📊 Summary Statistics:")
            print("-" * 50)
            print(f"  Mean semantic change: {df['semantic_change'].mean():.4f}")
            print(f"  Standard deviation:   {df['semantic_change'].std():.4f}")
            print(f"  Median change:        {df['semantic_change'].median():.4f}")
            
            # Load summary statistics
            summary_file = results_dir / "summary_statistics.csv"
            if summary_file.exists():
                summary_df = pd.read_csv(summary_file)
                print()
                print("📈 Detailed Statistics:")
                print("-" * 50)
                for col in summary_df.columns:
                    print(f"  {col}: {summary_df[col].iloc[0]}")
        
        else:
            print("❌ Semantic changes file not found")
            
    except Exception as e:
        print(f"❌ Error loading results: {e}")
    
    print()
    print("💡 Interpretation Tips:")
    print("-" * 50)
    print("• High change values suggest semantic shifts over time")
    print("• Low change values indicate stable meanings")
    print("• Function words (the, and, of) should show low changes")
    print("• Content words may show interesting historical patterns")
    print()
    print("🔬 Next Steps:")
    print("-" * 50)
    print("• Examine specific words of interest in detail")
    print("• Research historical context for high-change words")
    print("• Compare results with known linguistic changes")
    print("• Try different model parameters and compare results")

def check_models():
    """Check if trained models exist."""
    
    print("Checking for trained models...")
    print("-" * 30)
    
    model_files = [
        "word2vec_1900_20.model",
        "word2vec_1940_60.model",
        "word2vec_1900_20.pkl",
        "word2vec_1940_60.pkl"
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ {model_file}")
        else:
            print(f"❌ {model_file}")
    
    print()

if __name__ == "__main__":
    check_models()
    demonstrate_results()
