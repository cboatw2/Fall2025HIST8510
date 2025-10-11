#!/usr/bin/env python3
"""
Aligned Semantic Shift Visualization Script

This script creates visualizations showing how words and their semantic neighborhoods
change over time using Procrustes-aligned Word2Vec models. It includes error metrics
and statistical significance measures.

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
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlignedSemanticShiftVisualizer:
    """
    Class to visualize semantic shifts using Procrustes-aligned Word2Vec models.
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
        self.rotation_matrix = rotation_matrix
        self.common_words = valid_words
        
        # Filter alignment results to only include valid words
        self.alignment_results = self.alignment_results[
            self.alignment_results['word'].isin(valid_words)
        ].reset_index(drop=True)
        
        # Add missing columns if they don't exist
        if 'alignment_improvement' not in self.alignment_results.columns:
            self.alignment_results['alignment_improvement'] = (
                self.alignment_results['euclidean_distance_original'] - 
                self.alignment_results['euclidean_distance_aligned']
            )
        
        if 'alignment_improvement_pct' not in self.alignment_results.columns:
            self.alignment_results['alignment_improvement_pct'] = (
                (self.alignment_results['alignment_improvement'] / 
                 self.alignment_results['euclidean_distance_original']) * 100
            ).fillna(0)
        
        if 'is_significant' not in self.alignment_results.columns:
            threshold = self.alignment_results['semantic_change'].quantile(0.9)
            self.alignment_results['is_significant'] = (
                self.alignment_results['semantic_change'] > threshold
            )
        
        # Rename semantic_change to semantic_shift for consistency
        if 'semantic_change' in self.alignment_results.columns and 'semantic_shift' not in self.alignment_results.columns:
            self.alignment_results['semantic_shift'] = self.alignment_results['semantic_change']
    
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
        
        # Calculate error metrics
        self.calculate_error_metrics(common_words, vectors_1900_20, vectors_1940_60, 
                                   aligned_vectors_1900_20, rotation_matrix)
        
        # Store aligned vectors
        self.aligned_vectors_1900_20 = aligned_vectors_1900_20
        self.vectors_1940_60 = vectors_1940_60
        self.common_words = common_words
        self.rotation_matrix = rotation_matrix
    
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
    
    def calculate_error_metrics(self, words, vec1_orig, vec2, vec1_aligned, rotation_matrix):
        """Calculate comprehensive error metrics for alignment quality."""
        errors = []
        
        for i, word in enumerate(words):
            # Calculate distances
            dist_orig = np.linalg.norm(vec1_orig[i] - vec2[i])
            dist_aligned = np.linalg.norm(vec1_aligned[i] - vec2[i])
            
            # Calculate cosine similarities
            cos_sim_orig = np.dot(vec1_orig[i], vec2[i]) / (
                np.linalg.norm(vec1_orig[i]) * np.linalg.norm(vec2[i])
            )
            cos_sim_aligned = np.dot(vec1_aligned[i], vec2[i]) / (
                np.linalg.norm(vec1_aligned[i]) * np.linalg.norm(vec2[i])
            )
            
            # Calculate alignment improvement
            improvement = dist_orig - dist_aligned
            improvement_pct = (improvement / dist_orig) * 100 if dist_orig > 0 else 0
            
            # Calculate semantic shift magnitude
            semantic_shift = dist_aligned
            
            errors.append({
                'word': word,
                'euclidean_distance_original': dist_orig,
                'euclidean_distance_aligned': dist_aligned,
                'cosine_similarity_original': cos_sim_orig,
                'cosine_similarity_aligned': cos_sim_aligned,
                'alignment_improvement': improvement,
                'alignment_improvement_pct': improvement_pct,
                'semantic_shift': semantic_shift,
                'is_significant': semantic_shift > np.percentile([e['semantic_shift'] for e in errors], 90) if errors else False
            })
        
        self.alignment_results = pd.DataFrame(errors)
        self.alignment_results = self.alignment_results.sort_values('semantic_shift', ascending=False)
        
        # Calculate overall alignment quality
        self.alignment_quality = {
            'mean_improvement': self.alignment_results['alignment_improvement'].mean(),
            'mean_improvement_pct': self.alignment_results['alignment_improvement_pct'].mean(),
            'significant_shifts': len(self.alignment_results[self.alignment_results['semantic_shift'] > 
                                                           self.alignment_results['semantic_shift'].quantile(0.9)]),
            'total_words': len(self.alignment_results)
        }
        
        logger.info(f"Alignment quality metrics:")
        logger.info(f"  Mean improvement: {self.alignment_quality['mean_improvement']:.4f}")
        logger.info(f"  Mean improvement %: {self.alignment_quality['mean_improvement_pct']:.2f}%")
        logger.info(f"  Significant shifts: {self.alignment_quality['significant_shifts']}")
    
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
    
    def create_aligned_semantic_space_2d(self, words, n_neighbors=15):
        """Create a 2D semantic space using aligned vectors."""
        # Get vectors for target words and their neighbors
        all_words = []
        vectors_1900_aligned = []
        vectors_1940 = []
        
        for word in words:
            if word.lower() in self.common_vocab:
                # Find word index in common vocabulary
                word_idx = self.common_words.index(word.lower())
                
                # Add target word vectors
                all_words.append(word.lower())
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
                        vectors_1900_aligned.append(self.aligned_vectors_1900_20[sim_idx])
                        vectors_1940.append(self.vectors_1940_60[sim_idx])
                
                for sim_word, _ in similar_1940:
                    if sim_word in self.common_words and sim_word not in all_words:
                        sim_idx = self.common_words.index(sim_word)
                        all_words.append(sim_word)
                        vectors_1900_aligned.append(self.aligned_vectors_1900_20[sim_idx])
                        vectors_1940.append(self.vectors_1940_60[sim_idx])
        
        if len(vectors_1900_aligned) < 2:
            return None, None, None, None
        
        # Convert to numpy arrays
        vectors_1900_aligned = np.array(vectors_1900_aligned)
        vectors_1940 = np.array(vectors_1940)
        
        # Apply PCA to reduce to 2D (using combined space for consistent projection)
        combined_vectors = np.vstack([vectors_1900_aligned, vectors_1940])
        pca = PCA(n_components=2, random_state=42)
        combined_2d = pca.fit_transform(combined_vectors)
        
        # Split back into two periods
        n_points = len(vectors_1900_aligned)
        space_1900_aligned = combined_2d[:n_points]
        space_1940 = combined_2d[n_points:]
        
        return space_1900_aligned, space_1940, all_words, pca
    
    def plot_aligned_semantic_shift(self, word, output_dir="aligned_semantic_plots", 
                                   figsize=(15, 6), n_context_words=15):
        """Create an aligned semantic shift visualization with error metrics."""
        if word.lower() not in self.common_vocab:
            logger.warning(f"Word '{word}' not found in common vocabulary")
            return None
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get aligned semantic spaces
        space_1900_aligned, space_1940, words, pca = self.create_aligned_semantic_space_2d(
            [word], n_neighbors=n_context_words
        )
        
        if space_1900_aligned is None:
            logger.error(f"Could not create aligned semantic space for '{word}'")
            return None
        
        # Get error metrics for this word
        word_metrics = self.alignment_results[
            self.alignment_results['word'] == word.lower()
        ].iloc[0]
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Aligned semantic space
        self.plot_aligned_space(ax1, space_1900_aligned, space_1940, words, word, pca)
        
        # Plot 2: Error metrics
        self.plot_error_metrics(ax2, word_metrics)
        
        # Plot 3: Statistical significance
        self.plot_statistical_significance(ax3, word_metrics)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / f"aligned_semantic_shift_{word.lower()}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Aligned semantic shift plot saved to {output_file}")
        return output_file
    
    def plot_aligned_space(self, ax, space_1900_aligned, space_1940, words, target_word, pca):
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
        ax.scatter(space_1900_aligned[:, 0], space_1900_aligned[:, 1], 
                  c='lightblue', s=20, alpha=0.6, label='1900-20 (aligned)')
        ax.scatter(space_1940[:, 0], space_1940[:, 1], 
                  c='lightcoral', s=20, alpha=0.6, label='1940-60')
        
        # Highlight target word
        ax.scatter(space_1900_aligned[target_idx, 0], space_1900_aligned[target_idx, 1], 
                  c='blue', s=100, marker='o', edgecolors='blue', linewidth=2)
        ax.scatter(space_1940[target_idx, 0], space_1940[target_idx, 1], 
                  c='red', s=100, marker='o', edgecolors='red', linewidth=2)
        
        # Draw arrow showing the shift
        ax.annotate('', xy=(space_1940[target_idx, 0], space_1940[target_idx, 1]),
                   xytext=(space_1900_aligned[target_idx, 0], space_1900_aligned[target_idx, 1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        # Add labels for target word
        ax.annotate(f"{target_word} (1900-20)", 
                   (space_1900_aligned[target_idx, 0], space_1900_aligned[target_idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='blue')
        
        ax.annotate(f"{target_word} (1940-60)", 
                   (space_1940[target_idx, 0], space_1940[target_idx, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='red')
        
        # Add labels for context words (limit to avoid clutter)
        max_context_labels = min(10, len(words) - 1)  # Show up to 10 context words
        context_count = 0
        
        for i, (x, y) in enumerate(space_1900_aligned):
            if i != target_idx and context_count < max_context_labels:
                word = words[i]
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=7, color='blue', alpha=0.8)
                context_count += 1
        
        context_count = 0
        for i, (x, y) in enumerate(space_1940):
            if i != target_idx and context_count < max_context_labels:
                word = words[i]
                ax.annotate(word, (x, y), 
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=7, color='red', alpha=0.8)
                context_count += 1
        
        # Set title and remove axes
        ax.set_title(f"Aligned Semantic Space: {target_word.title()}", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(loc='upper right', fontsize=8)
    
    def plot_error_metrics(self, ax, word_metrics):
        """Plot error metrics for the word."""
        metrics = [
            ('Euclidean Distance (Original)', word_metrics['euclidean_distance_original']),
            ('Euclidean Distance (Aligned)', word_metrics['euclidean_distance_aligned']),
            ('Cosine Similarity (Original)', word_metrics['cosine_similarity_original']),
            ('Cosine Similarity (Aligned)', word_metrics['cosine_similarity_aligned']),
            ('Alignment Improvement', word_metrics['alignment_improvement']),
            ('Alignment Improvement %', word_metrics['alignment_improvement_pct'])
        ]
        
        labels = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        
        # Create bar plot
        bars = ax.bar(range(len(metrics)), values, color=['red', 'green', 'orange', 'blue', 'purple', 'brown'])
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title('Error Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        
        # Color code based on improvement
        for i, bar in enumerate(bars):
            if i in [1, 3, 4, 5]:  # Aligned metrics and improvements
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    def plot_statistical_significance(self, ax, word_metrics):
        """Plot statistical significance information."""
        # Calculate percentile rank
        semantic_shift = word_metrics['semantic_shift']
        percentile_rank = (self.alignment_results['semantic_shift'] <= semantic_shift).mean() * 100
        
        # Create text plot
        ax.text(0.5, 0.8, f"Semantic Shift: {semantic_shift:.4f}", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.6, f"Percentile Rank: {percentile_rank:.1f}%", 
                ha='center', va='center', fontsize=12)
        
        # Determine significance level
        if percentile_rank >= 90:
            significance = "Highly Significant"
            color = 'red'
        elif percentile_rank >= 75:
            significance = "Significant"
            color = 'orange'
        else:
            significance = "Not Significant"
            color = 'green'
        
        ax.text(0.5, 0.4, significance, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color)
        
        # Add context
        ax.text(0.5, 0.2, f"Out of {len(self.alignment_results)} words", 
                ha='center', va='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Statistical Significance', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def plot_multiple_aligned_words(self, words, output_dir="aligned_semantic_plots", 
                                   figsize=(20, 6), n_context_words=15):
        """Create aligned semantic shift visualizations for multiple words."""
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
            
            # Get aligned semantic spaces
            space_1900_aligned, space_1940, words_space, _ = self.create_aligned_semantic_space_2d(
                [word], n_neighbors=n_context_words
            )
            
            if space_1900_aligned is None:
                continue
            
            # Plot aligned space
            self.plot_aligned_space(ax, space_1900_aligned, space_1940, words_space, word, None)
        
        plt.tight_layout()
        
        # Save the plot
        words_str = "_".join([w.lower() for w in valid_words])
        output_file = output_path / f"aligned_semantic_shift_multiple_{words_str}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multiple aligned semantic shift plot saved to {output_file}")
        return output_file

def main():
    """Main function to create aligned semantic shift visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create aligned semantic shift visualizations')
    parser.add_argument('words', nargs='+', help='Words to analyze for semantic shift')
    parser.add_argument('--output-dir', default='aligned_semantic_plots', 
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
    visualizer = AlignedSemanticShiftVisualizer(
        args.model_1900, args.model_1940, args.alignment_results
    )
    
    if not visualizer.common_vocab:
        logger.error("No common vocabulary found between models")
        return
    
    logger.info(f"Common vocabulary size: {len(visualizer.common_vocab)}")
    
    # Create visualizations
    if len(args.words) == 1:
        # Single word visualization
        output_file = visualizer.plot_aligned_semantic_shift(
            args.words[0], args.output_dir, n_context_words=args.context_words
        )
        if output_file:
            logger.info(f"✅ Created aligned semantic shift plot for '{args.words[0]}' with {args.context_words} context words")
        else:
            logger.error(f"❌ Failed to create plot for '{args.words[0]}'")
    else:
        # Multiple words visualization
        output_file = visualizer.plot_multiple_aligned_words(
            args.words, args.output_dir, n_context_words=args.context_words
        )
        if output_file:
            logger.info(f"✅ Created aligned semantic shift plot for {len(args.words)} words with {args.context_words} context words each")
        else:
            logger.error("❌ Failed to create multiple words plot")

if __name__ == "__main__":
    main()
