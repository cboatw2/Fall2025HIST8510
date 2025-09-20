#!/usr/bin/env python3
"""
OCR with AI Correction Demo
============================

This script demonstrates how AI APIs can enhance OCR results for historical documents.
It performs OCR using Tesseract and then uses OpenAI's API to correct the text.

Usage:
    python ocr_with_ai_correction.py <image_path> [--api-key <key>]

Requirements:
    - OpenAI API key (set as environment variable OPENAI_API_KEY or pass via --api-key)
    - Tesseract installed on system
    - Historical document image file
"""

import argparse
import os
import sys
from pathlib import Path
import pytesseract
from PIL import Image
import openai
from openai import OpenAI


def perform_ocr(image_path):
    """
    Perform OCR on the given image using Tesseract.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from OCR
    """
    try:
        # Load and process the image
        image = Image.open(image_path)
        
        # Perform OCR with optimized settings for historical documents
        # PSM 3: Fully automatic page segmentation, but no OSD
        # This works well for historical documents
        text = pytesseract.image_to_string(
            image, 
            config='--psm 3'
        )
        
        return text.strip()
    
    except Exception as e:
        print(f"Error performing OCR: {e}")
        return None


def create_correction_prompt(ocr_text, document_type="historical document"):
    """
    Create a prompt for OpenAI to correct OCR text.
    
    Args:
        ocr_text (str): The raw OCR text
        document_type (str): Type of document being processed
        
    Returns:
        str: Formatted prompt for OpenAI
    """
    prompt = f"""Please correct the following OCR text from a {document_type}. 
The text may contain OCR errors, missing punctuation, or formatting issues.

IMPORTANT: You must process the ENTIRE document from beginning to end. Do not stop early or truncate the text.

Please:
1. Fix obvious OCR errors (like '0' instead of 'O', '1' instead of 'l', etc.)
2. Add appropriate punctuation and capitalization
3. Fix spacing and line breaks where needed
4. Preserve the original meaning and historical context
5. If a word is unclear, make your best guess based on context
6. Do not add any additional text to the original text
7. Do not delete any text from the original text unless you are sure it is an error and you are correcting a typo
8. Process EVERY line of the original text - do not skip any content

Original OCR text:
{ocr_text}

Corrected text:"""

    return prompt


def chunk_text(text, max_chunk_size=3000):
    """
    Split text into chunks that fit within token limits.
    
    Args:
        text (str): Text to chunk
        max_chunk_size (int): Maximum characters per chunk
        
    Returns:
        list: List of text chunks
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        if current_size + line_size > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks


def correct_text_with_ai(text, api_key):
    """
    Send OCR text to OpenAI API for correction.
    For large documents, this will chunk the text and process each chunk separately.
    
    Args:
        text (str): Raw OCR text to correct
        api_key (str): OpenAI API key
        
    Returns:
        str: Corrected text from AI
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Check if text is too large for single API call
        if len(text) > 3000:  # Rough estimate for token limit
            print("Large document detected. Processing in chunks...")
            chunks = chunk_text(text, max_chunk_size=3000)
            corrected_chunks = []
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                prompt = create_correction_prompt(chunk)
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert at correcting OCR text from historical documents. Focus on accuracy and preserving historical context. This is a chunk of a larger document - correct it completely but do not add introductory or concluding text."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
                
                corrected_chunk = response.choices[0].message.content.strip()
                corrected_chunks.append(corrected_chunk)
            
            # Combine all corrected chunks
            corrected_text = '\n'.join(corrected_chunks)
            return corrected_text
        
        else:
            # Process small document normally
            prompt = create_correction_prompt(text)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at correcting OCR text from historical documents. Focus on accuracy and preserving historical context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            corrected_text = response.choices[0].message.content.strip()
            return corrected_text
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def save_results(original_text, corrected_text, image_path, output_dir="ocr_results"):
    """
    Save both original and corrected text to files.
    
    Args:
        original_text (str): Original OCR text
        corrected_text (str): AI-corrected text
        image_path (str): Path to original image
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    # Save original OCR text
    original_file = Path(output_dir) / f"{base_name}_original_ocr.txt"
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(original_text)
    
    # Save corrected text
    corrected_file = Path(output_dir) / f"{base_name}_ai_corrected.txt"
    with open(corrected_file, 'w', encoding='utf-8') as f:
        f.write(corrected_text)
    
    print(f"\nResults saved:")
    print(f"  Original OCR: {original_file}")
    print(f"  AI Corrected: {corrected_file}")


def main():
    """Main function to run the OCR + AI correction workflow."""
    parser = argparse.ArgumentParser(description="OCR with AI correction for historical documents")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--output-dir", default="ocr_results", help="Directory to save results (default: ocr_results)")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    print(f"Processing image: {args.image_path}")
    print("=" * 50)
    
    # Step 1: Perform OCR
    print("Step 1: Performing OCR with Tesseract...")
    original_text = perform_ocr(args.image_path)
    
    if not original_text:
        print("Error: OCR failed to extract text.")
        sys.exit(1)
    
    print(f"OCR completed. Extracted {len(original_text)} characters.")
    print("\nOriginal OCR text:")
    print("-" * 30)
    print(original_text)
    print("-" * 30)
    
    # Step 2: Correct with AI
    print("\nStep 2: Sending to OpenAI for correction...")
    corrected_text = correct_text_with_ai(original_text, api_key)
    
    if not corrected_text:
        print("Error: AI correction failed.")
        sys.exit(1)
    
    print("AI correction completed.")
    print("\nCorrected text:")
    print("-" * 30)
    print(corrected_text)
    print("-" * 30)
    
    # Step 3: Save results
    print("\nStep 3: Saving results...")
    save_results(original_text, corrected_text, args.image_path, args.output_dir)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
