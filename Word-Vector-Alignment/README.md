# Word Vector Alignment Project

## Project Overview

### Data and Output Locations

- **Text Files**: Historical documents are stored in `txt/1900-20/` and `txt/1940-60/` directories
- **Trained Models**: Word2Vec models are saved as `.model` and `.pkl` files in the root directory
- **Visualizations**: Generated plots are saved to `visualizations/` directory
- **Results**: Analysis results are stored in `alignment_results/` directory

### Procrustes Alignment Algorithm

When you train word2vec models on different time periods, the word vectors end up in different coordinate systems - like having two maps of the same city but rotated differently. The Procrustes alignment algorithm finds the best way to rotate one map to match the other.

Think of it like this: if you have two word clouds where "king" and "queen" are close together in both, but the clouds are rotated differently, Procrustes finds the rotation that makes the clouds overlap as much as possible. This allows us to meaningfully compare how word meanings changed between time periods by ensuring we're looking at them from the same perspective.

## Setup Instructions

### 1. Check Python Version

```bash
# Check current Python version
python3 --version

# If you don't have Python 3.11, install it via Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

### 2. Create Virtual Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv word_vector_env

# Activate virtual environment
source word_vector_env/bin/activate  # On macOS/Linux
# OR
word_vector_env\Scripts\activate     # On Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Train Word2Vec Models
```bash
python train_word2vec_models.py
```

### 2. Align Vector Spaces
```bash
python align_word_vectors.py
```

### 3. Analyze Results
```bash
python demo_results.py
```

### 4. Create Visualizations

**Command line mode:**
```bash
python visualize_semantic_shifts.py academy absolute abnormal
```

### 5. Compare Aligned vs Unaligned Results
```bash
python compare_aligned_unaligned.py
```

### 6. Reset Project (if needed)
```bash
python reset_project.py
```