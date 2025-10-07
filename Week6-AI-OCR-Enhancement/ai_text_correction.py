#!/usr/bin/env python3
"""
AI Text Correction Script
=========================

A simple script that demonstrates how to use OpenAI's API to correct OCR text.
This script focuses just on the AI correction part, taking text input and improving it.

Usage:
    python ai_text_correction.py <text_file> [--api-key <key>]
"""

import argparse
import os
import sys
from pathlib import Path
from openai import OpenAI


def read_text_file(file_path):
    """
    Read text from a file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def create_correction_prompt(text):
    """
    Create a prompt for OpenAI to correct the text.
    
    Args:
        text (str): The text to correct
        
    Returns:
        str: Formatted prompt for OpenAI
    """
    prompt = f"""Please correct the following text that was extracted from a historical document using OCR. 
The text may contain OCR errors, missing punctuation, or formatting issues.

Please:
1. Fix obvious OCR errors (like '0' instead of 'O', '1' instead of 'l', etc.)
2. Add appropriate punctuation and capitalization
3. Fix spacing and line breaks where needed
4. Preserve the original meaning and historical context
5. If a word is unclear, make your best guess based on context

Text to correct:
{text}

Corrected text:"""

    return prompt


def correct_text_with_ai(text, api_key):
    """
    Send text to OpenAI API for correction.
    
    Args:
        text (str): Text to correct
        api_key (str): OpenAI API key
        
    Returns:
        str: Corrected text from AI
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create the correction prompt
        prompt = create_correction_prompt(text)
        
        # Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at correcting OCR text from historical documents. Focus on accuracy and preserving historical context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1  # Low temperature for more consistent corrections
        )
        
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def save_corrected_text(original_text, corrected_text, input_file):
    """
    Save the corrected text to a file.
    
    Args:
        original_text (str): Original text
        corrected_text (str): Corrected text
        input_file (str): Path to original input file
    """
    # Get base filename without extension
    base_name = Path(input_file).stem
    
    # Save corrected text
    output_file = f"{base_name}_ai_corrected.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corrected_text)
    
    print(f"Corrected text saved to: {output_file}")


def main():
    """Main function to run AI text correction."""
    parser = argparse.ArgumentParser(description="AI text correction for OCR output")
    parser.add_argument("text_file", help="Path to the text file to correct")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    
    args = parser.parse_args()
    
    # Check if text file exists
    if not os.path.exists(args.text_file):
        print(f"Error: Text file '{args.text_file}' not found.")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Set OPENAI_API_KEY environment variable or use --api-key option.")
        sys.exit(1)
    
    print(f"Processing text file: {args.text_file}")
    print("=" * 50)
    
    # Read the text file
    print("Reading text file...")
    original_text = read_text_file(args.text_file)
    
    if not original_text:
        print("Error: Failed to read text file.")
        sys.exit(1)
    
    print(f"Read {len(original_text)} characters from file.")
    print("\nOriginal text:")
    print("-" * 30)
    print(original_text)
    print("-" * 30)
    
    # Correct with AI
    print("\nSending to OpenAI for correction...")
    corrected_text = correct_text_with_ai(original_text, api_key)
    
    if not corrected_text:
        print("Error: AI correction failed.")
        sys.exit(1)
    
    print("AI correction completed.")
    print("\nCorrected text:")
    print("-" * 30)
    print(corrected_text)
    print("-" * 30)
    
    # Save results
    print("\nSaving corrected text...")
    save_corrected_text(original_text, corrected_text, args.text_file)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
