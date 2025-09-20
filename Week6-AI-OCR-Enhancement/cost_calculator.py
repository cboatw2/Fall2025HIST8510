#!/usr/bin/env python3
"""
Cost Calculator for OCR Methods
==============================

This script helps estimate costs for different OCR approaches
based on document size and processing requirements.

Usage:
    python cost_calculator.py [--tokens <number>] [--pages <number>]
"""

import argparse


def calculate_costs(tokens=None, pages=None):
    """
    Calculate estimated costs for different OCR methods.
    
    Args:
        tokens (int): Number of tokens to process
        pages (int): Number of pages to process
    """
    
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {
            "input": 0.0005,   # $0.50 per 1M tokens
            "output": 0.0015   # $1.50 per 1M tokens
        },
        "gpt-4o": {
            "input": 0.005,    # $5.00 per 1M tokens
            "output": 0.015    # $15.00 per 1M tokens
        }
    }
    
    # Typical token estimates
    tokens_per_page = {
        "small_document": 1000,    # ~1 page of text
        "medium_document": 3000,   # ~2-3 pages
        "large_document": 8000,    # ~5-10 pages
        "newspaper_page": 15000    # Full newspaper page
    }
    
    print("OCR COST CALCULATOR")
    print("=" * 50)
    
    if tokens:
        print(f"Processing {tokens:,} tokens:")
        print("-" * 30)
        
        # GPT-3.5-turbo costs
        gpt35_input_cost = (tokens / 1000) * pricing["gpt-3.5-turbo"]["input"]
        gpt35_output_cost = (tokens / 1000) * pricing["gpt-3.5-turbo"]["output"]
        gpt35_total = gpt35_input_cost + gpt35_output_cost
        
        # GPT-4o costs
        gpt4o_input_cost = (tokens / 1000) * pricing["gpt-4o"]["input"]
        gpt4o_output_cost = (tokens / 1000) * pricing["gpt-4o"]["output"]
        gpt4o_total = gpt4o_input_cost + gpt4o_output_cost
        
        print(f"GPT-3.5-turbo (OCR+AI): ${gpt35_total:.4f}")
        print(f"GPT-4o (Vision API):     ${gpt4o_total:.4f}")
        print(f"Cost difference:          ${gpt4o_total - gpt35_total:.4f}")
        
    elif pages:
        print(f"Processing {pages} page(s):")
        print("-" * 30)
        
        for doc_type, tokens_per in tokens_per_page.items():
            total_tokens = tokens_per * pages
            
            gpt35_cost = (total_tokens / 1000) * (pricing["gpt-3.5-turbo"]["input"] + pricing["gpt-3.5-turbo"]["output"])
            gpt4o_cost = (total_tokens / 1000) * (pricing["gpt-4o"]["input"] + pricing["gpt-4o"]["output"])
            
            print(f"{doc_type.replace('_', ' ').title()}:")
            print(f"  Tokens: {total_tokens:,}")
            print(f"  GPT-3.5-turbo: ${gpt35_cost:.4f}")
            print(f"  GPT-4o:        ${gpt4o_cost:.4f}")
            print()
    
    else:
        print("Example costs for different document sizes:")
        print("-" * 50)
        
        for doc_type, tokens_per in tokens_per_page.items():
            gpt35_cost = (tokens_per / 1000) * (pricing["gpt-3.5-turbo"]["input"] + pricing["gpt-3.5-turbo"]["output"])
            gpt4o_cost = (tokens_per / 1000) * (pricing["gpt-4o"]["input"] + pricing["gpt-4o"]["output"])
            
            print(f"{doc_type.replace('_', ' ').title()}:")
            print(f"  Tokens: {tokens_per:,}")
            print(f"  GPT-3.5-turbo: ${gpt35_cost:.4f}")
            print(f"  GPT-4o:        ${gpt4o_cost:.4f}")
            print(f"  Difference:    ${gpt4o_cost - gpt35_cost:.4f}")
            print()
    
    print("RECOMMENDATIONS:")
    print("-" * 50)
    print("• Use GPT-3.5-turbo (OCR+AI) for:")
    print("  - Large document collections")
    print("  - Budget-conscious projects")
    print("  - Documents with good image quality")
    print()
    print("• Use GPT-4o (Vision API) for:")
    print("  - Complex layouts and formatting")
    print("  - Poor quality images")
    print("  - When accuracy is critical")
    print("  - Single-step processing preferred")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Calculate OCR processing costs")
    parser.add_argument("--tokens", type=int, help="Number of tokens to process")
    parser.add_argument("--pages", type=int, help="Number of pages to process")
    
    args = parser.parse_args()
    
    calculate_costs(args.tokens, args.pages)


if __name__ == "__main__":
    main()


