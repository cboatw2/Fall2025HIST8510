# Named Entity Recognition (NER) with spaCy

This directory contains scripts for performing Named Entity Recognition on historical documents using spaCy.

## Files

- `ner_analysis_locations.py` - Extracts locations (GPE, LOC)
- `ner_analysis_people.py` - Extracts people (PERSON)
- `ner_analysis_custom_entityruler.py` - Custom entity extraction with EntityRuler
- `requirements.txt` - Python dependencies
- `setup.py` - Setup script for installation
- `reset.py` - Clean up results directory
- `APER_1942_February.txt` - Sample historical document

## Setup

### 1. Create Virtual Environment
```bash
python3 -m venv ner_env
source ner_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy English Model
```bash
python -m spacy download en_core_web_sm
```

## Available spaCy Models

- `en_core_web_sm` (small, fast) - Default, good for basic analysis
- `en_core_web_md` (medium) - Balanced speed and accuracy
- `en_core_web_lg` (large, accurate) - Better for historical texts

Download additional models:
```bash
python -m spacy download en_core_web_lg
```

## Usage

### Basic Usage (Default Model)
```bash
# Activate virtual environment
source ner_env/bin/activate

# Extract locations
python ner_analysis_locations.py APER_1942_February.txt

# Extract people
python ner_analysis_people.py APER_1942_February.txt

# Extract organizations
python ner_analysis_organizations.py APER_1942_February.txt
```

### Using Different Models
```bash
# Use large model for better accuracy
python ner_analysis_locations.py APER_1942_February.txt --model en_core_web_lg

# Use medium model
python ner_analysis_people.py APER_1942_February.txt --model en_core_web_md
```

## Output

All scripts save results to the `results/` directory as CSV files with entity details. 

## Clean Up

Reset the directory and remove all results:
```bash
python reset.py
```