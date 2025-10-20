#!/usr/bin/env python3
"""
Word Vector Alignment using Orthogonal Procrustes Analysis

This script aligns two word2vec models from different time periods using
orthogonal Procrustes analysis to enable meaningful comparison of semantic
changes over time.

Author: Lesson Plan for Digital Humanities Course
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_word2vec_model(model_path):
    """
    Load a Word2Vec model from file.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Word2Vec: Loaded Word2Vec model
    """
    try:
        # Try loading as gensim model first
        model = Word2Vec.load(model_path)
        logger.info(f"Loaded model from {model_path}")
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

def get_common_vocabulary(model1, model2, min_freq=5):
    """
    Get vocabulary that exists in both models with sufficient frequency.
    
    Args:
        model1 (Word2Vec): First word2vec model
        model2 (Word2Vec): Second word2vec model
        min_freq (int): Minimum frequency threshold
        
    Returns:
        list: List of common words
    """
    vocab1 = set(model1.wv.key_to_index.keys())
    vocab2 = set(model2.wv.key_to_index.keys())
    
    # Get intersection of vocabularies
    common_vocab = vocab1.intersection(vocab2)
    
    # Filter by frequency if available (gensim 4.x+)
    if hasattr(model1.wv, 'get_vecattr') and hasattr(model2.wv, 'get_vecattr'):
        try:
            common_vocab = [word for word in common_vocab 
                           if model1.wv.get_vecattr(word, 'count') >= min_freq and 
                              model2.wv.get_vecattr(word, 'count') >= min_freq]
        except:
            pass
    
    logger.info(f"Found {len(common_vocab)} common words")
    return sorted(list(common_vocab))

def extract_word_vectors(model, words):
    """
    Extract word vectors for given words from a model.
    
    Args:
        model (Word2Vec): Word2vec model
        words (list): List of words to extract vectors for
        
    Returns:
        numpy.ndarray: Matrix of word vectors
    """
    vectors = []
    valid_words = []
    
    for word in words:
        try:
            vector = model.wv[word]
            vectors.append(vector)
            valid_words.append(word)
        except KeyError:
            logger.warning(f"Word '{word}' not found in model")
    
    return np.array(vectors), valid_words

def orthogonal_procrustes_alignment(X, Y):
    """
    Perform orthogonal Procrustes analysis to align two sets of vectors.
    
    Args:
        X (numpy.ndarray): Source vectors (n_samples, n_features)
        Y (numpy.ndarray): Target vectors (n_samples, n_features)
        
    Returns:
        tuple: (aligned_X, rotation_matrix)
    """
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

def analyze_semantic_changes(model1, model2, aligned_model1, common_words, top_n=10):
    """
    Analyze semantic changes between the two time periods.
    
    Args:
        model1 (Word2Vec): Original model from first period
        model2 (Word2Vec): Model from second period
        aligned_model1 (numpy.ndarray): Aligned vectors from first period
        common_words (list): List of common words
        top_n (int): Number of top changes to show
        
    Returns:
        pandas.DataFrame: DataFrame with semantic change analysis
    """
    changes = []
    
    for i, word in enumerate(common_words):
        # Get original vectors
        vec1_orig = model1.wv[word]
        vec2 = model2.wv[word]
        vec1_aligned = aligned_model1[i]
        
        # Calculate distances
        dist_orig = np.linalg.norm(vec1_orig - vec2)
        dist_aligned = np.linalg.norm(vec1_aligned - vec2)
        
        # Calculate cosine similarity
        cos_sim_orig = np.dot(vec1_orig, vec2) / (np.linalg.norm(vec1_orig) * np.linalg.norm(vec2))
        cos_sim_aligned = np.dot(vec1_aligned, vec2) / (np.linalg.norm(vec1_aligned) * np.linalg.norm(vec2))
        
        changes.append({
            'word': word,
            'euclidean_distance_original': dist_orig,
            'euclidean_distance_aligned': dist_aligned,
            'cosine_similarity_original': cos_sim_orig,
            'cosine_similarity_aligned': cos_sim_aligned,
            'semantic_change': dist_aligned  # Use aligned distance as measure of change
        })
    
    df = pd.DataFrame(changes)
    df = df.sort_values('semantic_change', ascending=False)
    
    return df

def visualize_alignment_results(df, output_dir="alignment_results"):
    """
    Create visualizations of the alignment results.
    
    Args:
        df (pandas.DataFrame): Results from semantic change analysis
        output_dir (str): Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results
    df.to_csv(output_path / "semantic_changes.csv", index=False)
    logger.info(f"Detailed results saved to {output_path / 'semantic_changes.csv'}")
    
    # Create summary statistics
    summary = {
        'total_words_analyzed': len(df),
        'mean_semantic_change': df['semantic_change'].mean(),
        'std_semantic_change': df['semantic_change'].std(),
        'words_with_high_change': len(df[df['semantic_change'] > df['semantic_change'].quantile(0.9)]),
        'words_with_low_change': len(df[df['semantic_change'] < df['semantic_change'].quantile(0.1)])
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / "summary_statistics.csv", index=False)
    logger.info(f"Summary statistics saved to {output_path / 'summary_statistics.csv'}")
    
    return summary

def main():
    """Main function to perform word vector alignment."""
    
    logger.info("Starting Word Vector Alignment Analysis")
    logger.info("=" * 50)
    
    # Load models
    model_1900_20 = load_word2vec_model("word2vec_1900_20.model")
    model_1940_60 = load_word2vec_model("word2vec_1940_60.model")
    
    if model_1900_20 is None or model_1940_60 is None:
        logger.error("Failed to load one or both models. Please run train_word2vec_models.py first.")
        return
    
    logger.info(f"Model 1900-20 vocabulary size: {len(model_1900_20.wv)}")
    logger.info(f"Model 1940-60 vocabulary size: {len(model_1940_60.wv)}")
    
    # Get common vocabulary
    common_words = get_common_vocabulary(model_1900_20, model_1940_60, min_freq=5)
    
    if len(common_words) < 10:
        logger.error("Not enough common words found. Try reducing min_freq or check your data.")
        return
    
    logger.info(f"Analyzing {len(common_words)} common words")
    
    # Extract word vectors
    logger.info("Extracting word vectors...")
    vectors_1900_20, valid_words_1900_20 = extract_word_vectors(model_1900_20, common_words)
    vectors_1940_60, valid_words_1940_60 = extract_word_vectors(model_1940_60, common_words)
    
    # Ensure we have the same words in both sets
    assert valid_words_1900_20 == valid_words_1940_60, "Word order mismatch"
    
    logger.info(f"Extracted vectors for {len(valid_words_1900_20)} words")
    
    # Perform orthogonal Procrustes alignment
    logger.info("Performing orthogonal Procrustes alignment...")
    aligned_vectors_1900_20, rotation_matrix = orthogonal_procrustes_alignment(
        vectors_1900_20, vectors_1940_60
    )
    
    logger.info("Alignment completed")
    logger.info(f"Rotation matrix shape: {rotation_matrix.shape}")
    
    # Analyze semantic changes
    logger.info("Analyzing semantic changes...")
    results_df = analyze_semantic_changes(
        model_1900_20, model_1940_60, aligned_vectors_1900_20, valid_words_1900_20
    )
    
    # Display top changes
    logger.info("Top 10 words with largest semantic changes:")
    top_changes = results_df.head(10)
    for _, row in top_changes.iterrows():
        logger.info(f"  {row['word']}: {row['semantic_change']:.4f}")
    
    # Display words with smallest changes
    logger.info("Top 10 words with smallest semantic changes:")
    bottom_changes = results_df.tail(10)
    for _, row in bottom_changes.iterrows():
        logger.info(f"  {row['word']}: {row['semantic_change']:.4f}")
    
    # Create visualizations and save results
    summary = visualize_alignment_results(results_df)
    
    logger.info("=" * 50)
    logger.info("ALIGNMENT ANALYSIS COMPLETED")
    logger.info("=" * 50)
    logger.info("Results saved to 'alignment_results/' directory:")
    logger.info("  - semantic_changes.csv: Detailed analysis for each word")
    logger.info("  - summary_statistics.csv: Overall statistics")
    logger.info(f"Total words analyzed: {summary['total_words_analyzed']}")
    logger.info(f"Mean semantic change: {summary['mean_semantic_change']:.4f}")
    logger.info(f"Words with high change: {summary['words_with_high_change']}")
    logger.info(f"Words with low change: {summary['words_with_low_change']}")

if __name__ == "__main__":
    main()
