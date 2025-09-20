# Week 6: AI-Enhanced OCR for Historical Documents

This demo shows how AI APIs can significantly improve OCR results for historical documents by combining traditional OCR (Tesseract) with modern AI text correction (OpenAI).

## Files

- `ocr_with_ai_correction.py` - Complete workflow (OCR + AI correction)
- `simple_ocr.py` - Basic OCR-only script for comparison
- `vision_api_ocr.py` - Direct image-to-text using OpenAI Vision API
- `cleanup_results.py` - Clean up generated .txt files (Python version)
- `cleanup.sh` - Clean up generated .txt files (Shell version)
- `requirements.txt` - Python dependencies

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Get OpenAI API Key

1. Sign up at https://platform.openai.com/
2. Create an API key
3. Set as environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### Basic OCR Only

```bash
python simple_ocr.py path/to/your/image.jpg
```

### OCR + AI Correction

```bash
python ocr_with_ai_correction.py path/to/your/image.jpg
```

Or with API key as argument:
```bash
python ocr_with_ai_correction.py path/to/your/image.jpg --api-key your-api-key
```

### Vision API (Direct Image-to-Text)

```bash
python vision_api_ocr.py path/to/your/image.jpg
```

### Clean Up Results

Remove all generated .txt files to start fresh:

```bash
# Python version (with dry-run option)
python cleanup_results.py --dry-run    # See what would be deleted
python cleanup_results.py              # Actually delete files

# Shell version (simpler)
./cleanup.sh --dry-run                 # See what would be deleted  
./cleanup.sh                           # Actually delete files
```
