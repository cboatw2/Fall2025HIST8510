#!/usr/bin/env python3
"""
Multilingual Vision API OCR Demo
================================

This script demonstrates how to use OpenAI's Vision API to transcribe
text from images in various languages and scripts.

Usage:
    python multilingual_vision_ocr.py <image_path> [--language <lang>] [--api-key <key>]

Supported languages:
    auto, english, spanish, french, german, chinese, japanese, korean, 
    arabic, hindi, tamil, bengali, telugu, marathi, gujarati, punjabi
"""

import argparse
import os
import sys
from pathlib import Path
import base64
from openai import OpenAI


def encode_image(image_path):
    """
    Encode image to base64 for OpenAI Vision API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def create_multilingual_prompt(language="auto"):
    """
    Create a prompt optimized for multilingual text transcription.
    
    Args:
        language (str): Target language or "auto" for automatic detection
        
    Returns:
        str: Formatted prompt for OpenAI Vision API
    """
    
    language_instructions = {
        "auto": "The text may be in any language. Please detect the language and transcribe accordingly.",
        "english": "The text is in English.",
        "spanish": "The text is in Spanish (Español).",
        "french": "The text is in French (Français).",
        "german": "The text is in German (Deutsch).",
        "chinese": "The text is in Chinese (中文). Transcribe Chinese characters exactly as written.",
        "japanese": "The text is in Japanese (日本語). Transcribe Hiragana, Katakana, and Kanji exactly as written.",
        "korean": "The text is in Korean (한국어). Transcribe Hangul characters exactly as written.",
        "arabic": "The text is in Arabic (العربية). Transcribe Arabic script from right to left exactly as written.",
        "hindi": "The text is in Hindi (हिन्दी). Transcribe Devanagari script exactly as written.",
        "tamil": "The text is in Tamil (தமிழ்). Transcribe Tamil script exactly as written.",
        "bengali": "The text is in Bengali (বাংলা). Transcribe Bengali script exactly as written.",
        "telugu": "The text is in Telugu (తెలుగు). Transcribe Telugu script exactly as written.",
        "marathi": "The text is in Marathi (मराठी). Transcribe Devanagari script exactly as written.",
        "gujarati": "The text is in Gujarati (ગુજરાતી). Transcribe Gujarati script exactly as written.",
        "punjabi": "The text is in Punjabi (ਪੰਜਾਬੀ). Transcribe Gurmukhi script exactly as written."
    }
    
    lang_instruction = language_instructions.get(language.lower(), language_instructions["auto"])
    
    prompt = f"""Please transcribe all the text visible in this image. {lang_instruction}

CRITICAL INSTRUCTIONS:
1. Extract ALL text from the image, preserving the original layout and structure
2. Maintain line breaks and paragraph structure exactly as they appear
3. Do not add any commentary, interpretation, or translation
4. If text is unclear or partially obscured, transcribe what you can see
5. Preserve original spelling and formatting in the original language
6. Include headers, titles, dates, and all visible text elements
7. For non-Latin scripts (Arabic, Chinese, Tamil, etc.), transcribe characters exactly as written
8. Do not translate the text - only transcribe it
9. If you cannot read a character, use [?] to indicate uncertainty
10. Return only the transcribed text in its original language

IMPORTANT: Even if the text appears difficult to read or in an unfamiliar script, please attempt to transcribe it. Do not refuse to transcribe based on language or script type."""

    return prompt


def transcribe_multilingual_image(image_path, api_key, language="auto"):
    """
    Send image to OpenAI Vision API for multilingual text transcription.
    
    Args:
        image_path (str): Path to the image file
        api_key (str): OpenAI API key
        language (str): Target language or "auto"
        
    Returns:
        tuple: (transcribed_text, usage_info)
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode the image
        base64_image = encode_image(image_path)
        if not base64_image:
            return None, None
        
        # Determine image format
        image_format = Path(image_path).suffix.lower()
        if image_format not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            print(f"Warning: {image_format} may not be supported by Vision API")
        
        # Create language-specific prompt
        prompt = create_multilingual_prompt(language)
        
        # Send request to OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format[1:]};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Extract usage information
        usage = response.usage
        usage_info = {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        }
        
        transcribed_text = response.choices[0].message.content.strip()
        return transcribed_text, usage_info
        
    except Exception as e:
        print(f"Error calling OpenAI Vision API: {e}")
        return None, None


def calculate_cost(usage_info, model="gpt-4o"):
    """
    Calculate estimated cost based on token usage.
    
    Args:
        usage_info (dict): Token usage information
        model (str): Model used for the API call
        
    Returns:
        dict: Cost breakdown
    """
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-4o": {
            "input": 0.005,    # $5.00 per 1M tokens
            "output": 0.015    # $15.00 per 1M tokens
        }
    }
    
    input_cost = (usage_info['prompt_tokens'] / 1000) * pricing[model]["input"]
    output_cost = (usage_info['completion_tokens'] / 1000) * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model
    }


def print_usage_summary(usage_info, cost_info):
    """
    Print token usage and cost summary to terminal.
    
    Args:
        usage_info (dict): Token usage information
        cost_info (dict): Cost breakdown
    """
    print("\n" + "=" * 50)
    print("OPENAI VISION API USAGE SUMMARY")
    print("=" * 50)
    print(f"Model: {cost_info['model']}")
    print(f"Prompt Tokens:  {usage_info['prompt_tokens']:,}")
    print(f"Output Tokens:  {usage_info['completion_tokens']:,}")
    print(f"Total Tokens:   {usage_info['total_tokens']:,}")
    print("-" * 50)
    print(f"Input Cost:     ${cost_info['input_cost']:.4f}")
    print(f"Output Cost:    ${cost_info['output_cost']:.4f}")
    print(f"Total Cost:     ${cost_info['total_cost']:.4f}")
    print("=" * 50)


def save_transcription(text, image_path, language, output_dir="multilingual_results"):
    """
    Save transcribed text to a file.
    
    Args:
        text (str): Transcribed text to save
        image_path (str): Path to original image
        language (str): Language used for transcription
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    # Save transcribed text
    output_file = Path(output_dir) / f"{base_name}_{language}_transcription.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Multilingual transcription saved to: {output_file}")


def main():
    """Main function to run the multilingual Vision API OCR workflow."""
    parser = argparse.ArgumentParser(description="Multilingual OCR using OpenAI Vision API")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--language", default="auto", 
                       help="Target language (auto, english, spanish, french, german, chinese, japanese, korean, arabic, hindi, tamil, etc.)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY environment variable)")
    parser.add_argument("--output-dir", default="multilingual_results", 
                       help="Directory to save results (default: multilingual_results)")
    
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
    print(f"Language: {args.language}")
    print("=" * 50)
    
    # Transcribe with multilingual Vision API
    print("Step 1: Sending image to OpenAI Vision API...")
    transcribed_text, usage_info = transcribe_multilingual_image(args.image_path, api_key, args.language)
    
    if not transcribed_text:
        print("Error: Vision API transcription failed.")
        sys.exit(1)
    
    print("Multilingual Vision API transcription completed.")
    print(f"Transcribed {len(transcribed_text)} characters.")
    print("\nTranscribed text:")
    print("-" * 30)
    print(transcribed_text)
    print("-" * 30)
    
    # Calculate and display usage/cost
    if usage_info:
        cost_info = calculate_cost(usage_info, "gpt-4o")
        print_usage_summary(usage_info, cost_info)
    
    # Save results
    print("\nStep 2: Saving results...")
    save_transcription(transcribed_text, args.image_path, args.language, args.output_dir)
    
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()

