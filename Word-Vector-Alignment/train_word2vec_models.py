#!/usr/bin/env python3
"""
Word2Vec Model Training Script for Historical Text Analysis

This script trains separate word2vec models on text documents from two different
time periods (1900-20 and 1940-60) to analyze semantic changes over time.

Author: Lesson Plan for Digital Humanities Course
"""

import os
import re
import pickle
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.utils import simple_preprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_text(text):
    """
    Preprocess text by cleaning and tokenizing using gensim's simple_preprocess.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        list: List of tokens
    """
    # Use gensim's simple_preprocess for consistent tokenization
    tokens = simple_preprocess(text, min_len=2, max_len=50)
    
    return tokens

def load_text_files(directory):
    """
    Load and preprocess all text files from a directory.
    
    Args:
        directory (str): Path to directory containing text files
        
    Returns:
        list: List of tokenized sentences
    """
    sentences = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory {directory} does not exist")
        return sentences
    
    # Get all .txt files in the directory
    txt_files = list(directory_path.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files in {directory}")
    
    for file_path in txt_files:
        logger.info(f"Processing {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Split content into sentences (simple approach)
            # Split on periods, exclamation marks, and question marks
            raw_sentences = re.split(r'[.!?]+', content)
            
            for sentence in raw_sentences:
                if sentence.strip():  # Skip empty sentences
                    tokens = preprocess_text(sentence)
                    if len(tokens) > 2:  # Only keep sentences with more than 2 words
                        sentences.append(tokens)
                        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Total sentences processed: {len(sentences)}")
    return sentences

def train_word2vec_model(sentences, model_name, vector_size=100, window=5, min_count=5, workers=4):
    """
    Train a Word2Vec model on the provided sentences using gensim.
    
    Args:
        sentences (list): List of tokenized sentences
        model_name (str): Name for the model file
        vector_size (int): Size of word vectors
        window (int): Maximum distance between current and predicted word
        min_count (int): Minimum count of words to be included in vocabulary
        workers (int): Number of worker threads
        
    Returns:
        Word2Vec: Trained Word2Vec model
    """
    logger.info(f"Training Word2Vec model: {model_name}")
    logger.info(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}")
    
    # Train the model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10,  # Number of iterations over the corpus
        sg=1,  # Skip-gram model (1) vs CBOW (0)
        negative=5,  # Number of negative samples
        hs=0,  # Hierarchical softmax (0 = disabled)
        sample=1e-3,  # Downsampling threshold
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001,  # Final learning rate
        seed=42  # Random seed for reproducibility
    )
    
    # Save the model
    model_path = f"{model_name}.model"
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Also save as pickle for easier loading
    pickle_path = f"{model_name}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model also saved as pickle: {pickle_path}")
    
    # Print some statistics
    logger.info(f"Vocabulary size: {len(model.wv)}")
    logger.info(f"Total words in corpus: {model.corpus_total_words}")
    
    return model

def main():
    """Main function to train word2vec models for both time periods."""
    
    # Define paths
    base_dir = Path(__file__).parent
    txt_dir = base_dir / "txt"
    period_1900_20 = txt_dir / "1900-20"
    period_1940_60 = txt_dir / "1940-60"
    
    # Model parameters
    vector_size = 100
    window = 5
    min_count = 5
    workers = 4
    
    logger.info("Starting Word2Vec model training for historical text analysis")
    
    # Train model for 1900-20 period
    logger.info("=" * 50)
    logger.info("TRAINING MODEL FOR 1900-20 PERIOD")
    logger.info("=" * 50)
    
    sentences_1900_20 = load_text_files(period_1900_20)
    if sentences_1900_20:
        model_1900_20 = train_word2vec_model(
            sentences_1900_20, 
            "word2vec_1900_20",
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        
        # Show some example similar words
        logger.info("Example similar words for 'education':")
        try:
            similar_words = model_1900_20.wv.most_similar('education', topn=5)
            for word, score in similar_words:
                logger.info(f"  {word}: {score:.3f}")
        except KeyError:
            logger.info("  'education' not found in vocabulary")
    else:
        logger.error("No sentences found for 1900-20 period")
    
    # Train model for 1940-60 period
    logger.info("=" * 50)
    logger.info("TRAINING MODEL FOR 1940-60 PERIOD")
    logger.info("=" * 50)
    
    sentences_1940_60 = load_text_files(period_1940_60)
    if sentences_1940_60:
        model_1940_60 = train_word2vec_model(
            sentences_1940_60, 
            "word2vec_1940_60",
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        
        # Show some example similar words
        logger.info("Example similar words for 'education':")
        try:
            similar_words = model_1940_60.wv.most_similar('education', topn=5)
            for word, score in similar_words:
                logger.info(f"  {word}: {score:.3f}")
        except KeyError:
            logger.info("  'education' not found in vocabulary")
    else:
        logger.error("No sentences found for 1940-60 period")
    
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 50)
    logger.info("Models saved:")
    logger.info("  - word2vec_1900_20.model")
    logger.info("  - word2vec_1900_20.pkl")
    logger.info("  - word2vec_1940_60.model")
    logger.info("  - word2vec_1940_60.pkl")
    logger.info("Next step: Run alignment script to align the models")

if __name__ == "__main__":
    main()
