#!/usr/bin/env python3
"""
Cleanup Script for OCR Results
==============================

This script removes all generated .txt files from OCR processing,
allowing you to start fresh with clean result directories.

Usage:
    python cleanup_results.py [--dry-run]

Options:
    --dry-run    Show what would be deleted without actually deleting
"""

import argparse
import os
import sys
from pathlib import Path


def find_txt_files():
    """
    Find all .txt files in the current directory and subdirectories.
    
    Returns:
        list: List of Path objects for .txt files
    """
    txt_files = []
    
    # Files to exclude from cleanup (important project files)
    exclude_files = {
        "requirements.txt",
        "README.txt",
        "LICENSE.txt"
    }
    
    # Directories where OCR results are typically saved
    result_dirs = [
        "ocr_results",
        "vision_results", 
        "comparison_results"
    ]
    
    # Check each result directory
    for dir_name in result_dirs:
        if Path(dir_name).exists():
            for txt_file in Path(dir_name).glob("*.txt"):
                txt_files.append(txt_file)
    
    # Also check for any .txt files in the root directory (excluding important files)
    for txt_file in Path(".").glob("*.txt"):
        if txt_file.name not in exclude_files:
            txt_files.append(txt_file)
    
    return txt_files


def cleanup_files(dry_run=False):
    """
    Delete all .txt files found in result directories.
    
    Args:
        dry_run (bool): If True, only show what would be deleted
    """
    txt_files = find_txt_files()
    
    if not txt_files:
        print("No .txt files found to clean up.")
        return
    
    print(f"Found {len(txt_files)} .txt file(s) to clean up:")
    print("-" * 50)
    
    total_size = 0
    deleted_count = 0
    
    for txt_file in txt_files:
        try:
            file_size = txt_file.stat().st_size
            total_size += file_size
            
            if dry_run:
                print(f"[DRY RUN] Would delete: {txt_file} ({file_size:,} bytes)")
            else:
                txt_file.unlink()
                print(f"Deleted: {txt_file} ({file_size:,} bytes)")
                deleted_count += 1
                
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    print("-" * 50)
    
    if dry_run:
        print(f"[DRY RUN] Would delete {len(txt_files)} files ({total_size:,} bytes total)")
    else:
        print(f"Successfully deleted {deleted_count} files ({total_size:,} bytes total)")


def cleanup_directories(dry_run=False):
    """
    Remove empty result directories.
    
    Args:
        dry_run (bool): If True, only show what would be deleted
    """
    result_dirs = [
        "ocr_results",
        "vision_results", 
        "comparison_results"
    ]
    
    print("\nChecking for empty result directories:")
    print("-" * 50)
    
    for dir_name in result_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                # Check if directory is empty
                if not any(dir_path.iterdir()):
                    if dry_run:
                        print(f"[DRY RUN] Would remove empty directory: {dir_name}")
                    else:
                        dir_path.rmdir()
                        print(f"Removed empty directory: {dir_name}")
                else:
                    remaining_files = list(dir_path.glob("*"))
                    print(f"Directory {dir_name} still contains {len(remaining_files)} file(s)")
            except Exception as e:
                print(f"Error processing directory {dir_name}: {e}")


def main():
    """Main function to run the cleanup."""
    parser = argparse.ArgumentParser(description="Clean up OCR result files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    print("OCR Results Cleanup Script")
    print("=" * 50)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will actually be deleted")
        print("=" * 50)
    
    # Confirm before proceeding (unless dry run)
    if not args.dry_run:
        response = input("\nAre you sure you want to delete all .txt files? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            sys.exit(0)
    
    # Clean up files
    cleanup_files(args.dry_run)
    
    # Clean up empty directories
    cleanup_directories(args.dry_run)
    
    print("\nCleanup completed!")


if __name__ == "__main__":
    main()
