#!/bin/bash
# Quick cleanup script for OCR results
# Usage: ./cleanup.sh [--dry-run]

echo "OCR Results Cleanup Script"
echo "=========================="

if [[ "$1" == "--dry-run" ]]; then
    echo "DRY RUN MODE - No files will actually be deleted"
    echo "=================================================="
    
    echo "Files that would be deleted:"
    find ocr_results vision_results comparison_results -name "*.txt" 2>/dev/null | while read file; do
        echo "  $file ($(wc -c < "$file") bytes)"
    done
    
    echo ""
    echo "Empty directories that would be removed:"
    for dir in ocr_results vision_results comparison_results; do
        if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
            echo "  $dir"
        fi
    done
else
    echo "This will delete all .txt files in result directories."
    echo "Press Enter to continue or Ctrl+C to cancel..."
    read
    
    echo "Deleting .txt files..."
    find ocr_results vision_results comparison_results -name "*.txt" -delete 2>/dev/null
    
    echo "Removing empty directories..."
    for dir in ocr_results vision_results comparison_results; do
        if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
            rmdir "$dir"
            echo "Removed empty directory: $dir"
        fi
    done
    
    echo "Cleanup completed!"
fi
