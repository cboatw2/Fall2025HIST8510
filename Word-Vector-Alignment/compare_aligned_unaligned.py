#!/usr/bin/env python3
"""
Comparison Visualization: Aligned vs Unaligned Semantic Spaces

This script creates side-by-side visualizations showing the difference between
unaligned and Procrustes-aligned semantic spaces, demonstrating why alignment
is crucial for meaningful semantic shift analysis.

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
from scipy.linalg import orthogonal_procrustes
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlignedUnalignedComparison:
    """
    Class to compare aligned vs unaligned semantic spaces.
    """
    
    def __init__(self, model_1900_20_path, model_1940_60_path, alignment_results_path=None):
        """Initialize with paths to models and alignment results."""
        self.model_1900_20 = self.load_model(model_1900_20_path)
        self.model_1940_60 = self.load_model(model_1940_60_path)
        self.common_vocab = self.get_common_vocabulary()
        
        # Load alignment results if available
        self.alignment_results = None
        if alignment_results_path and Path(alignment_results_path).exists():
            self.alignment_results = pd.read_csv(alignment_results_path)
            logger.info(f"Loaded alignment results from {alignment_results_path}")
            # Extract common words from alignment results
            self.common_words = self.alignment_results['word'].tolist()
            # Extract aligned vectors from models
            self.extract_aligned_vectors()
        else:
            # Perform alignment if not already done
            logger.info("Performing Procrustes alignment...")
            self.perform_alignment()
        
    def load_model(self, model_path):
        """Load a gensim Word2Vec model."""
        try:
            model = Word2Vec.load(model_path)
            logger.info(f"Loaded gensim model from {model_path}")
            return model
        except:
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
    
    def extract_aligned_vectors(self):
        """Extract aligned vectors from the models using the alignment results."""
        # Filter common words to only include those present in both models
        valid_words = []
        for word in self.common_words:
            if (word in self.model_1900_20.wv.key_to_index and 
                word in self.model_1940_60.wv.key_to_index):
                valid_words.append(word)
        
        logger.info(f"Found {len(valid_words)} valid words in both models")
        
        # Extract vectors for valid words
        vectors_1900_20 = np.array([self.model_1900_20.wv[word] for word in valid_words])
        vectors_1940_60 = np.array([self.model_1940_60.wv[word] for word in valid_words])
        
        # Perform alignment to get aligned vectors
        aligned_vectors_1900_20, rotation_matrix = self.orthogonal_procrustes_alignment(
            vectors_1900_20, vectors_1940_60
        )
        
        # Store aligned vectors and update common words
        self.aligned_vectors_1900_20 = aligned_vectors_1900_20
        self.vectors_1940_60 = vectors_1940_60
        self.vectors_1900_20_original = vectors_1900_20  # Store original unaligned vectors
        self.rotation_matrix = rotation_matrix
        self.common_words = valid_words
        
        # Filter alignment results to only include valid words
        self.alignment_results = self.alignment_results[
            self.alignment_results['word'].isin(valid_words)
        ].reset_index(drop=True)
    
    def orthogonal_procrustes_alignment(self, X, Y):
        """Perform orthogonal Procrustes analysis to align two sets of vectors."""
        # Center the vectors
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Compute the rotation matrix using orthogonal Procrustes
        R, _ = orthogonal_procrustes(Y_centered, X_centered)
        
        # Apply rotation to align X to Y
        X_aligned = X_centered @ R.T
        
        # Add back the mean of Y to maintain the target space
        X_aligned = X_aligned + np.mean(Y, axis=0)
        
        return X_aligned, R
    
    def perform_alignment(self):
        """Perform Procrustes alignment and calculate error metrics."""
        common_words = sorted(list(self.common_vocab))
        
        # Extract vectors
        vectors_1900_20 = np.array([self.model_1900_20.wv[word] for word in common_words])
        vectors_1940_60 = np.array([self.model_1940_60.wv[word] for word in common_words])
        
        # Perform alignment
        aligned_vectors_1900_20, rotation_matrix = self.orthogonal_procrustes_alignment(
            vectors_1900_20, vectors_1940_60
        )
        
        # Store vectors
        self.aligned_vectors_1900_20 = aligned_vectors_1900_20
        self.vectors_1940_60 = vectors_1940_60
        self.vectors_1900_20_original = vectors_1900_20
        self.common_words = common_words
        self.rotation_matrix = rotation_matrix
    
    def get_similar_words(self, model, word, topn=10):
        """Get most similar words to the target word using gensim."""
        if word.lower() not in model.wv.key_to_index:
            return []
        
        try:
            return model.wv.most_similar(word.lower(), topn=topn)
        except:
            return []
    
    def create_comparison_space_2d(self, words, n_neighbors=15):
        """Create 2D spaces for both aligned and unaligned vectors."""
        # Get vectors for target words and their neighbors
        all_words = []
        vectors_1900_original = []
        vectors_1900_aligned = []
        vectors_1940 = []
        
        for word in words:
            if word.lower() in self.common_words:
                # Find word index in common vocabulary
                word_idx = self.common_words.index(word.lower())
                
                # Add target word vectors
                all_words.append(word.lower())
                vectors_1900_original.append(self.vectors_1900_20_original[word_idx])
                vectors_1900_aligned.append(self.aligned_vectors_1900_20[word_idx])
                vectors_1940.append(self.vectors_1940_60[word_idx])
                
                # Add similar words from both periods
                similar_1900 = self.get_similar_words(self.model_1900_20, word.lower(), topn=n_neighbors)
                similar_1940 = self.get_similar_words(self.model_1940_60, word.lower(), topn=n_neighbors)
                
                # Add similar words that exist in common vocabulary AND alignment results
                for sim_word, _ in similar_1900:
                    if sim_word in self.common_words and sim_word not in all_words:
                        sim_idx = self.common_words.index(sim_word)
                        all_words.append(sim_word)
                        vectors_1900_original.append(self.vectors_1900_20_original[sim_idx])
                        vectors_1900_aligned.append(self.aligned_vectors_1900_20[sim_idx])
                        vectors_1940.append(self.vectors_1940_60[sim_idx])
                
                for sim_word, _ in similar_1940:
                    if sim_word in self.common_words and sim_word not in all_words:
                        sim_idx = self.common_words.index(sim_word)
                        all_words.append(sim_word)
                        vectors_1900_original.append(self.vectors_1900_20_original[sim_idx])
                        vectors_1900_aligned.append(self.aligned_vectors_1900_20[sim_idx])
                        vectors_1940.append(self.vectors_1940_60[sim_idx])
        
        if len(vectors_1900_original) < 2:
            return None, None, None, None, None
        
        # Convert to numpy arrays
        vectors_1900_original = np.array(vectors_1900_original)
        vectors_1900_aligned = np.array(vectors_1900_aligned)
        vectors_1940 = np.array(vectors_1940)
        
        # Apply PCA to reduce to 2D (using combined space for consistent projection)
        combined_vectors = np.vstack([vectors_1900_original, vectors_1900_aligned, vectors_1940])
        pca = PCA(n_components=2, random_state=42)
        combined_2d = pca.fit_transform(combined_vectors)
        
        # Split back into three periods
        n_points = len(vectors_1900_original)
        space_1900_original = combined_2d[:n_points]
        space_1900_aligned = combined_2d[n_points:2*n_points]
        space_1940 = combined_2d[2*n_points:]
        
        return space_1900_original, space_1900_aligned, space_1940, all_words, pca
    
    def plot_comparison(self, word, output_dir="comparison_plots", figsize=(20, 8), n_context_words=15):
        """Create a comparison visualization showing aligned vs unaligned spaces."""
        if word.lower() not in self.common_words:
            logger.warning(f"Word '{word}' not found in common vocabulary")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get comparison spaces
        space_1900_original, space_1900_aligned, space_1940, words, pca = self.create_comparison_space_2d(
            [word], n_neighbors=n_context_words
        )
        
        if space_1900_original is None:
            logger.error(f"Could not create comparison space for '{word}'")
            return None
        
        # Get error metrics for this word
        word_metrics = None
        if self.alignment_results is not None:
            word_data = self.alignment_results[self.alignment_results['word'] == word.lower()]
            if not word_data.empty:
                word_metrics = word_data.iloc[0]
        
        # Create the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Unaligned semantic space
        self.plot_unaligned_space(ax1, space_1900_original, space_1940, words, word, "Unaligned")
        
        # Plot 2: Aligned semantic space
        self.plot_aligned_space(ax2, space_1900_aligned, space_1940, words, word, "Aligned")
        
        # Plot 3: Distance comparison
        self.plot_distance_comparison(ax3, word_metrics, word)
        
        # Plot 4: Alignment quality metrics
        self.plot_alignment_quality(ax4, word_metrics, word)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / f"comparison_{word.lower()}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_file}")
        return output_file
    
    def plot_unaligned_space(self, ax, space_1900, space_1940, words, target_word, title_suffix):
        """Plot the unaligned semantic space."""
        # Find target word position
        target_idx = None
        for i, w in enumerate(words):
            if w.lower() == target_word.lower():
                target_idx = i
                break
        
        if target_idx is None:
            return
        
        # Plot all words as small dots
        ax.scatter(space_1900[:, 0], space_1900[:, 1], 
                  c='lightblue', s=20, alpha=0.6, label='1900-20 (unaligned)')
        ax.scatter(space_1940[:, 0], space_1940[:, 1], 
                  c='lightcoral', s=20, alpha=0.6, label='1940-60')
        
        # Highlight target word
        ax.scatter(space_1900[target_idx, 0], space_1900[target_idx, 1], 
                  c='blue', s=100, marker='o', edgecolors='blue', linewidth=2)
        ax.scatter(space_1940[target_idx, 0], space_1940[target_idx, 1], 
                  c='red', s=100, marker='o', edgecolors='red', linewidth=2)
        
        # Draw arrow showing the shift
        ax.annotate('', xy=(space_1940[target_idx, 0], space_1940[target_idx, 1]),
                   xytext=(space_1900[target_idx, 0], space_1900[target_idx, 1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Add labels for target word
        ax.annotate(f"{target_word} (1900-20)", 
                   (space_1900[target_idx, 0], space_1900[target_idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='blue')
        
        ax.annotate(f"{target_word} (1940-60)", 
                   (space_1940[target_idx, 0], space_1940[target_idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red')
        
        # Add labels for context words (limit to avoid clutter)
        max_context_labels = min(8, len(words) - 1)  # Show up to 8 context words
        context_count = 0
        
        for i, (x, y) in enumerate(space_1900):
            if i != target_idx and context_count < max_context_labels:
                word = words[i]
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=6, color='blue', alpha=0.8)
                context_count += 1
        
        context_count = 0
        for i, (x, y) in enumerate(space_1940):
            if i != target_idx and context_count < max_context_labels:
                word = words[i]
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=6, color='red', alpha=0.8)
                context_count += 1
        
        # Set title and remove axes
        ax.set_title(f"Semantic Space: {target_word.title()} ({title_suffix})", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(loc='upper right', fontsize=8)
    
    def plot_aligned_space(self, ax, space_1900, space_1940, words, target_word, title_suffix):
        """Plot the aligned semantic space."""
        # Find target word position
        target_idx = None
        for i, w in enumerate(words):
            if w.lower() == target_word.lower():
                target_idx = i
                break
        
        if target_idx is None:
            return
        
        # Plot all words as small dots
        ax.scatter(space_1900[:, 0], space_1900[:, 1], 
                  c='lightblue', s=20, alpha=0.6, label='1900-20 (aligned)')
        ax.scatter(space_1940[:, 0], space_1940[:, 1], 
                  c='lightcoral', s=20, alpha=0.6, label='1940-60')
        
        # Highlight target word
        ax.scatter(space_1900[target_idx, 0], space_1900[target_idx, 1], 
                  c='blue', s=100, marker='o', edgecolors='blue', linewidth=2)
        ax.scatter(space_1940[target_idx, 0], space_1940[target_idx, 1], 
                  c='red', s=100, marker='o', edgecolors='red', linewidth=2)
        
        # Draw arrow showing the shift
        ax.annotate('', xy=(space_1940[target_idx, 0], space_1940[target_idx, 1]),
                   xytext=(space_1900[target_idx, 0], space_1900[target_idx, 1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Add labels for target word
        ax.annotate(f"{target_word} (1900-20)", 
                   (space_1900[target_idx, 0], space_1900[target_idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='blue')
        
        ax.annotate(f"{target_word} (1940-60)", 
                   (space_1940[target_idx, 0], space_1940[target_idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red')
        
        # Add labels for context words (limit to avoid clutter)
        max_context_labels = min(8, len(words) - 1)  # Show up to 8 context words
        context_count = 0
        
        for i, (x, y) in enumerate(space_1900):
            if i != target_idx and context_count < max_context_labels:
                word = words[i]
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=6, color='blue', alpha=0.8)
                context_count += 1
        
        context_count = 0
        for i, (x, y) in enumerate(space_1940):
            if i != target_idx and context_count < max_context_labels:
                word = words[i]
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=6, color='red', alpha=0.8)
                context_count += 1
        
        # Set title and remove axes
        ax.set_title(f"Semantic Space: {target_word.title()} ({title_suffix})", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(loc='upper right', fontsize=8)
    
    def plot_distance_comparison(self, ax, word_metrics, word):
        """Plot distance comparison between aligned and unaligned."""
        if word_metrics is None:
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center', fontsize=12)
            ax.set_title('Distance Comparison', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Extract distances
        dist_original = word_metrics.get('euclidean_distance_original', 0)
        dist_aligned = word_metrics.get('euclidean_distance_aligned', 0)
        
        # Create bar plot
        categories = ['Original\n(Unaligned)', 'Aligned']
        distances = [dist_original, dist_aligned]
        colors = ['red', 'green']
        
        bars = ax.bar(categories, distances, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, distance in zip(bars, distances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{distance:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Euclidean Distance Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance')
        
        # Add improvement annotation
        improvement = dist_original - dist_aligned
        improvement_pct = (improvement / dist_original) * 100 if dist_original > 0 else 0
        ax.text(0.5, max(distances) * 0.8, f'Improvement: {improvement:.3f}\n({improvement_pct:.1f}%)', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def plot_alignment_quality(self, ax, word_metrics, word):
        """Plot alignment quality metrics."""
        if word_metrics is None:
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center', fontsize=12)
            ax.set_title('Alignment Quality', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # Extract metrics
        cos_sim_original = word_metrics.get('cosine_similarity_original', 0)
        cos_sim_aligned = word_metrics.get('cosine_similarity_aligned', 0)
        semantic_shift = word_metrics.get('semantic_change', 0)
        
        # Create text display
        ax.text(0.1, 0.8, f"Cosine Similarity (Original): {cos_sim_original:.3f}", 
                ha='left', va='center', fontsize=11)
        ax.text(0.1, 0.6, f"Cosine Similarity (Aligned): {cos_sim_aligned:.3f}", 
                ha='left', va='center', fontsize=11)
        ax.text(0.1, 0.4, f"Semantic Shift Magnitude: {semantic_shift:.3f}", 
                ha='left', va='center', fontsize=11)
        
        # Determine significance
        if semantic_shift > 0.5:
            significance = "High Change"
            color = 'red'
        elif semantic_shift > 0.3:
            significance = "Moderate Change"
            color = 'orange'
        else:
            significance = "Low Change"
            color = 'green'
        
        ax.text(0.1, 0.2, f"Significance: {significance}", 
                ha='left', va='center', fontsize=11, fontweight='bold', color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Alignment Quality Metrics', fontsize=12, fontweight='bold')
        ax.axis('off')

def main():
    """Main function to create comparison visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create aligned vs unaligned comparison visualizations')
    parser.add_argument('words', nargs='+', help='Words to analyze for semantic shift')
    parser.add_argument('--output-dir', default='comparison_plots', 
                       help='Output directory for plots')
    parser.add_argument('--model-1900', default='word2vec_1900_20.model',
                       help='Path to 1900-20 gensim model')
    parser.add_argument('--model-1940', default='word2vec_1940_60.model',
                       help='Path to 1940-60 gensim model')
    parser.add_argument('--alignment-results', default='alignment_results/semantic_changes.csv',
                       help='Path to alignment results CSV')
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
    visualizer = AlignedUnalignedComparison(
        args.model_1900, args.model_1940, args.alignment_results
    )
    
    if not visualizer.common_vocab:
        logger.error("No common vocabulary found between models")
        return
    
    logger.info(f"Common vocabulary size: {len(visualizer.common_vocab)}")
    
    # Create visualizations
    for word in args.words:
        output_file = visualizer.plot_comparison(
            word, args.output_dir, n_context_words=args.context_words
        )
        if output_file:
            logger.info(f"✅ Created comparison plot for '{word}' with {args.context_words} context words")
        else:
            logger.error(f"❌ Failed to create plot for '{word}'")

if __name__ == "__main__":
    main()
