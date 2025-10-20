#!/usr/bin/env python3
"""
Interactive Semantic Shift Visualization Tool for Gensim Models

This tool provides an interactive interface for creating semantic shift visualizations
using gensim Word2Vec models.

Author: Lesson Plan for Digital Humanities Course
"""

import sys
from pathlib import Path
from visualize_gensim_shifts import GensimSemanticShiftVisualizer

def get_user_input():
    """Get words to analyze from user input."""
    print("Semantic Shift Visualization Tool")
    print("=" * 50)
    print("This tool will create visualizations showing how words and their")
    print("semantic neighborhoods change between 1900-20 and 1940-60 periods.")
    print()
    
    # Check if model files exist
    model_1900 = "word2vec_1900_20.model"
    model_1940 = "word2vec_1940_60.model"
    
    if not Path(model_1900).exists() or not Path(model_1940).exists():
        print("❌ Model files not found!")
        print("Please run the following commands first:")
        print("  1. python train_word2vec_models.py")
        print("  2. python align_word_vectors.py")
        return None, None, None, None
    
    # Get words from user
    print("Enter words to analyze (separated by commas):")
    print("Examples: education, physical, training, health")
    print()
    
    user_input = input("Words: ").strip()
    if not user_input:
        print("No words provided. Exiting.")
        return None, None, None, None
    
    # Parse words
    words = [word.strip().lower() for word in user_input.split(',')]
    words = [word for word in words if word]  # Remove empty strings
    
    if not words:
        print("No valid words provided. Exiting.")
        return None, None, None, None
    
    print(f"\nAnalyzing {len(words)} word(s): {', '.join(words)}")
    
    # Ask for number of context words
    print("\nHow many context words would you like to show around each target word?")
    print("(More context words = richer semantic neighborhoods, but may be cluttered)")
    print("Default: 15")
    
    context_input = input("Number of context words [15]: ").strip()
    if context_input:
        try:
            n_context = int(context_input)
            if n_context < 1:
                n_context = 15
        except ValueError:
            n_context = 15
    else:
        n_context = 15
    
    print(f"Using {n_context} context words per target word")
    
    return words, model_1900, model_1940, n_context

def show_available_words(visualizer, limit=50):
    """Show some available words from the vocabulary."""
    print(f"\nAvailable words in vocabulary (showing first {limit}):")
    print("-" * 50)
    
    vocab_list = sorted(list(visualizer.common_vocab))
    for i, word in enumerate(vocab_list[:limit]):
        print(f"{word:<15}", end='')
        if (i+1) % 5 == 0:
            print()
    
    if len(vocab_list) > limit:
        print(f"\n... and {len(vocab_list) - limit} more words")
    print()

def main():
    """Main interactive function."""
    # Get user input
    words, model_1900, model_1940, n_context = get_user_input()
    if words is None:
        return
    
    # Create visualizer
    print("Loading models...")
    visualizer = GensimSemanticShiftVisualizer(model_1900, model_1940)
    
    if not visualizer.common_vocab:
        print("❌ No common vocabulary found between models")
        return
    
    print(f"✅ Loaded models with {len(visualizer.common_vocab)} common words")
    
    # Show available words if requested
    show_vocab = input("\nWould you like to see available words in the vocabulary? (y/n): ").strip().lower()
    if show_vocab in ['y', 'yes']:
        show_available_words(visualizer)
    
    # Filter words that exist in vocabulary
    available_words = [w for w in words if w.lower() in visualizer.common_vocab]
    missing_words = [w for w in words if w.lower() not in visualizer.common_vocab]
    
    if missing_words:
        print(f"⚠️  Words not found in vocabulary: {', '.join(missing_words)}")
    
    if not available_words:
        print("❌ None of the provided words are in the vocabulary")
        return
    
    print(f"✅ Analyzing {len(available_words)} available word(s): {', '.join(available_words)}")
    
    # Create visualizations
    print("\nCreating semantic shift visualizations...")
    
    try:
        if len(available_words) == 1:
            # Single word visualization
            output_file = visualizer.plot_semantic_shift(available_words[0], n_context_words=n_context)
            if output_file:
                print(f"✅ Created plot: {output_file}")
            else:
                print(f"❌ Failed to create plot for '{available_words[0]}'")
        else:
            # Multiple words visualization
            output_file = visualizer.plot_multiple_words(available_words, n_context_words=n_context)
            if output_file:
                print(f"✅ Created plot: {output_file}")
            else:
                print("❌ Failed to create multiple words plot")
        
        print("\n🎉 Visualization complete!")
        print("Check the 'semantic_shift_plots' directory for your plots.")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
