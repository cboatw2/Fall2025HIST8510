# Week 9: N-grams and Jaccard Similarity Analysis

## Files in this folder:
- `ngram_jaccard_analysis.py` - Main analysis script
- `data_philly.csv` - Location data from LGBTQ guidebooks
- `requirements.txt` - Dependencies (minimal - uses only Python standard library)
- `similarity_results.csv` - Output file (created after running analysis)

## Setup Instructions:

### First Time Setup (Create New Virtual Environment):

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis:**
   ```bash
   python ngram_jaccard_analysis.py
   ```

5. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Returning to Existing Virtual Environment:

1. **Activate the existing virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the analysis:**
   ```bash
   python ngram_jaccard_analysis.py
   ```

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

## What it does:
Analyzes location titles using traditional n-grams and Jaccard similarity to find similar locations across different guidebook editions (e.g., "Joe's Bar" vs "Joes Bar").

## Output:
Creates `similarity_results.csv` with similarity scores and location details.