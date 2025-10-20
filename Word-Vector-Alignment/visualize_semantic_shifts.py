#!/usr/bin/env python3
"""
Semantic Shift Visualization Script for Gensim Word2Vec Models

This script creates visualizations showing how words and their semantic neighborhoods
change over time using gensim Word2Vec models.

Author: Lesson Plan for Digital Humanities Course
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GensimSemanticShiftVisualizer:
    """
    Class to visualize semantic shifts of words over time using gensim models.
    """
    
    def __init__(self, model_1900_20_path, model_1940_60_path):
        """Initialize with paths to the trained gensim models."""
        self.model_1900_20 = self.load_model(model_1900_20_path)
        self.model_1940_60 = self.load_model(model_1940_60_path)
        self.common_vocab = self.get_common_vocabulary()
        
    def load_model(self, model_path):
        """Load a gensim Word2Vec model."""
        try:
            # Try loading as gensim model first
            model = Word2Vec.load(model_path)
            logger.info(f"Loaded gensim model from {model_path}")
            return model
        except:
            # Try loading as pickle
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model from pickle {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                return None
    
    def get_common_vocabulary(self):
        """Get vocabulary that exists in both models."""
        if self.model_1900_20 is None or self.model_1940_60 is None:
            return set()
        
        vocab1 = set(self.model_1900_20.wv.key_to_index.keys())
        vocab2 = set(self.model_1940_60.wv.key_to_index.keys())
        return vocab1.intersection(vocab2)
    
    def get_word_vector(self, model, word):
        """Get word vector from gensim model."""
        if word.lower() in model.wv.key_to_index:
            return model.wv[word.lower()]
        return None
    
    def get_similar_words(self, model, word, topn=10):
        """Get most similar words to the target word using gensim."""
        if word.lower() not in model.wv.key_to_index:
            return []
        
        try:
            return model.wv.most_similar(word.lower(), topn=topn)
        except:
            return []
    
    def create_semantic_space_2d(self, words, model, n_neighbors=15):
        """Create a 2D semantic space for visualization."""
        # Get vectors for target words and their neighbors
        all_words = []
        vectors = []
        
        for word in words:
            if word.lower() in model.wv.key_to_index:
                # Add target word
                all_words.append(word.lower())
                vectors.append(self.get_word_vector(model, word.lower()))
                
                # Add similar words
                similar_words = self.get_similar_words(model, word.lower(), topn=n_neighbors)
                for sim_word, sim_score in similar_words:
                    if sim_word not in all_words and sim_word in model.wv.key_to_index:
                        all_words.append(sim_word)
                        vectors.append(self.get_word_vector(model, sim_word))
        
        if len(vectors) < 2:
            return None, None, None
        
        # Convert to numpy array
        vectors = np.array(vectors)
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2, random_state=42)
        vectors_2d = pca.fit_transform(vectors)
        
        return vectors_2d, all_words, pca
    
    def plot_semantic_shift(self, word, output_dir="semantic_shift_plots", figsize=(12, 8), n_context_words=15):
        """Create a semantic shift visualization for a single word."""
        if word.lower() not in self.common_vocab:
            logger.warning(f"Word '{word}' not found in common vocabulary")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get semantic spaces for both time periods
        space_1900_20, words_1900_20, pca_1900_20 = self.create_semantic_space_2d([word], self.model_1900_20, n_neighbors=n_context_words)
        space_1940_60, words_1940_60, pca_1940_60 = self.create_semantic_space_2d([word], self.model_1940_60, n_neighbors=n_context_words)
        
        if space_1900_20 is None or space_1940_60 is None:
            logger.error(f"Could not create semantic space for '{word}'")
            return None
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1900-20 period
        self.plot_semantic_space(ax1, space_1900_20, words_1900_20, word, "1900-20", pca_1900_20)
        
        # Plot 1940-60 period
        self.plot_semantic_space(ax2, space_1940_60, words_1940_60, word, "1940-60", pca_1940_60)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / f"semantic_shift_{word.lower()}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Semantic shift plot saved to {output_file}")
        return output_file
    
    def plot_semantic_space(self, ax, space_2d, words, target_word, period, pca):
        """Plot a single semantic space."""
        # Find target word position
        target_idx = None
        for i, w in enumerate(words):
            if w.lower() == target_word.lower():
                target_idx = i
                break
        
        if target_idx is None:
            return
        
        # Plot all words as small grey dots
        ax.scatter(space_2d[:, 0], space_2d[:, 1], c='lightgrey', s=20, alpha=0.6)
        
        # Highlight target word
        ax.scatter(space_2d[target_idx, 0], space_2d[target_idx, 1], 
                  c='black', s=100, marker='o', edgecolors='black', linewidth=2)
        
        # Add labels for target word and nearby words
        for i, (x, y) in enumerate(space_2d):
            word = words[i]
            if word.lower() == target_word.lower():
                # Target word in bold
                ax.annotate(f"{word} ({period})", (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='black')
            else:
                # Similar words in grey
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=8, color='grey', alpha=0.7)
        
        # Set title and remove axes
        ax.set_title(f"Semantic Space: {target_word.title()} ({period})", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def plot_multiple_words(self, words, output_dir="semantic_shift_plots", figsize=(15, 5), n_context_words=15):
        """Create semantic shift visualizations for multiple words."""
        if not words:
            logger.error("No words provided")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Filter words that exist in common vocabulary
        valid_words = [w for w in words if w.lower() in self.common_vocab]
        if not valid_words:
            logger.error("None of the provided words found in common vocabulary")
            return None
        
        # Create subplots
        n_words = len(valid_words)
        fig, axes = plt.subplots(1, n_words, figsize=figsize)
        if n_words == 1:
            axes = [axes]
        
        for i, word in enumerate(valid_words):
            ax = axes[i]
            
            # Get semantic spaces for both periods
            space_1900_20, words_1900_20, _ = self.create_semantic_space_2d([word], self.model_1900_20, n_neighbors=n_context_words)
            space_1940_60, words_1940_60, _ = self.create_semantic_space_2d([word], self.model_1940_60, n_neighbors=n_context_words)
            
            if space_1900_20 is None or space_1940_60 is None:
                continue
            
            # Create combined plot showing both periods
            self.plot_combined_semantic_shift(ax, space_1900_20, words_1900_20, 
                                            space_1940_60, words_1940_60, word)
        
        plt.tight_layout()
        
        # Save the plot
        words_str = "_".join([w.lower() for w in valid_words])
        output_file = output_path / f"semantic_shift_multiple_{words_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multiple semantic shift plot saved to {output_file}")
        return output_file
    
    def plot_combined_semantic_shift(self, ax, space_1900_20, words_1900_20, 
                                   space_1940_60, words_1940_60, target_word):
        """Plot combined semantic shift showing both time periods."""
        # Find target word positions
        target_idx_1900 = None
        target_idx_1940 = None
        
        for i, w in enumerate(words_1900_20):
            if w.lower() == target_word.lower():
                target_idx_1900 = i
                break
        
        for i, w in enumerate(words_1940_60):
            if w.lower() == target_word.lower():
                target_idx_1940 = i
                break
        
        if target_idx_1900 is None or target_idx_1940 is None:
            return
        
        # Plot 1900-20 words in light blue
        ax.scatter(space_1900_20[:, 0], space_1900_20[:, 1], 
                  c='lightblue', s=15, alpha=0.5, label='1900-20')
        
        # Plot 1940-60 words in light coral
        ax.scatter(space_1940_60[:, 0], space_1940_60[:, 1], 
                  c='lightcoral', s=15, alpha=0.5, label='1940-60')
        
        # Highlight target word in both periods
        ax.scatter(space_1900_20[target_idx_1900, 0], space_1900_20[target_idx_1900, 1], 
                  c='blue', s=100, marker='o', edgecolors='blue', linewidth=2)
        ax.scatter(space_1940_60[target_idx_1940, 0], space_1940_60[target_idx_1940, 1], 
                  c='red', s=100, marker='o', edgecolors='red', linewidth=2)
        
        # Draw arrow showing the shift
        ax.annotate('', xy=(space_1940_60[target_idx_1940, 0], space_1940_60[target_idx_1940, 1]),
                   xytext=(space_1900_20[target_idx_1900, 0], space_1900_20[target_idx_1900, 1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Add labels
        ax.annotate(f"{target_word} (1900-20)", 
                   (space_1900_20[target_idx_1900, 0], space_1900_20[target_idx_1900, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='blue')
        
        ax.annotate(f"{target_word} (1940-60)", 
                   (space_1940_60[target_idx_1940, 0], space_1940_60[target_idx_1940, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red')
        
        # Add neighbor labels (show more context words)
        for i, (x, y) in enumerate(space_1900_20):
            if i != target_idx_1900 and i < 10:  # Show first 10 neighbors
                ax.annotate(words_1900_20[i], (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=7, color='blue', alpha=0.8)
        
        for i, (x, y) in enumerate(space_1940_60):
            if i != target_idx_1940 and i < 10:  # Show first 10 neighbors
                ax.annotate(words_1940_60[i], (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=7, color='red', alpha=0.8)
        
        # Set title and remove axes
        ax.set_title(f"Semantic Shift: {target_word.title()}", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(loc='upper right', fontsize=8)

def main():
    """Main function to create semantic shift visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create semantic shift visualizations using gensim models')
    parser.add_argument('words', nargs='+', help='Words to analyze for semantic shift')
    parser.add_argument('--output-dir', default='semantic_shift_plots', 
                       help='Output directory for plots')
    parser.add_argument('--model-1900', default='word2vec_1900_20.model',
                       help='Path to 1900-20 gensim model')
    parser.add_argument('--model-1940', default='word2vec_1940_60.model',
                       help='Path to 1940-60 gensim model')
    parser.add_argument('--context-words', type=int, default=15,
                       help='Number of context words to show around each target word')
    
    args = parser.parse_args()
    
    # Check if model files exist
    if not Path(args.model_1900).exists():
        logger.error(f"Model file not found: {args.model_1900}")
        logger.info("Please run train_word2vec_models.py first")
        return
    
    if not Path(args.model_1940).exists():
        logger.error(f"Model file not found: {args.model_1940}")
        logger.info("Please run train_word2vec_models.py first")
        return
    
    # Create visualizer
    visualizer = GensimSemanticShiftVisualizer(args.model_1900, args.model_1940)
    
    if not visualizer.common_vocab:
        logger.error("No common vocabulary found between models")
        return
    
    logger.info(f"Common vocabulary size: {len(visualizer.common_vocab)}")
    
    # Create visualizations
    if len(args.words) == 1:
        # Single word visualization
        output_file = visualizer.plot_semantic_shift(args.words[0], args.output_dir, n_context_words=args.context_words)
        if output_file:
            logger.info(f"✅ Created semantic shift plot for '{args.words[0]}' with {args.context_words} context words")
        else:
            logger.error(f"❌ Failed to create plot for '{args.words[0]}'")
    else:
        # Multiple words visualization
        output_file = visualizer.plot_multiple_words(args.words, args.output_dir, n_context_words=args.context_words)
        if output_file:
            logger.info(f"✅ Created semantic shift plot for {len(args.words)} words with {args.context_words} context words each")
        else:
            logger.error("❌ Failed to create multiple words plot")

if __name__ == "__main__":
    main()
