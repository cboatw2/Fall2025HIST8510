#!/usr/bin/env python3
"""
Simple OCR Script
=================

A basic script to perform OCR on historical document images using Tesseract.
This is a simpler version for students to understand the OCR process.

Usage:
    python simple_ocr.py <image_path>
"""

import sys
import pytesseract
from PIL import Image
from pathlib import Path


def perform_ocr(image_path):
    """
    Perform OCR on the given image using Tesseract.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from OCR
    """
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Perform OCR with settings optimized for historical documents
        text = pytesseract.image_to_string(
            image, 
            config='--psm 3'  # Fully automatic page segmentation
        )
        
        return text.strip()
    
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return None


def save_ocr_text(text, image_path, output_dir="ocr_results"):
    """
    Save OCR text to a file.
    
    Args:
        text (str): OCR text to save
        image_path (str): Path to original image
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    # Save OCR text
    output_file = Path(output_dir) / f"{base_name}_ocr.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"OCR text saved to: {output_file}")


def main():
    """Main function to run OCR."""
    if len(sys.argv) != 2:
        print("Usage: python simple_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image file exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    print("=" * 50)
    
    # Perform OCR
    print("Performing OCR with Tesseract...")
    text = perform_ocr(image_path)
    
    if not text:
        print("Error: OCR failed to extract text.")
        sys.exit(1)
    
    print(f"OCR completed. Extracted {len(text)} characters.")
    print("\nExtracted text:")
    print("-" * 30)
    print(text)
    print("-" * 30)
    
    # Save results
    print("\nSaving results...")
    save_ocr_text(text, image_path)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
