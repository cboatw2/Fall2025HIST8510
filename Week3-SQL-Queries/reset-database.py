#!/usr/bin/env python3
"""
Reset Script - Restore Week3-SQL-Queries to Original State
History 8510 - Clemson University

This script resets the folder to its original state before any scripts were run.
Use this to clean up before starting fresh or to reset after class demos.
"""

import os
import shutil

def reset_to_original():
    """Reset the folder to its original state"""
    
    print("=== RESETTING TO ORIGINAL STATE ===")
    print("History 8510 - Week 3 SQL Queries")
    print("=" * 40)
    
    # Files to remove (generated during script execution)
    files_to_remove = [
        'sc_gay_guides.db',
        'sc_geographic_expansion_results.csv',
        'geographic_expansion_results.csv'
    ]
    
    # Remove generated files
    print("\n🗑️  Removing generated files...")
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"   ✓ Removed: {file}")
        else:
            print(f"   - Not found: {file}")
    
    print("\n✅ Reset complete! Generated files removed.")
    print("\n📋 Current files:")
    
    # Show current directory contents
    current_files = [f for f in os.listdir('.') if f.endswith(('.py', '.csv', '.db'))]
    for file in sorted(current_files):
        print(f"   - {file}")

if __name__ == "__main__":
    # Ask for confirmation before resetting
    print("⚠️  WARNING: This will remove all generated files!")
    print("This action cannot be undone.")
    
    response = input("\nAre you sure you want to reset? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        reset_to_original()
    else:
        print("Reset cancelled.")
