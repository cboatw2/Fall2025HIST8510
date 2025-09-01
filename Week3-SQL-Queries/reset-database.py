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
    
    # Rename numbered scripts back to original names
    print("\n🔄 Renaming scripts back to original names...")
    
    rename_mapping = {
        '01-create-sc-database.py': 'create-db-sc.py',
        '02-import-sc-data.py': 'import-data-sc.py',
        '03-analyze-sc-geography.py': 'simple_geographic_analysis.py'
    }
    
    for numbered_name, original_name in rename_mapping.items():
        if os.path.exists(numbered_name):
            os.rename(numbered_name, original_name)
            print(f"   ✓ Renamed: {numbered_name} → {original_name}")
        else:
            print(f"   - Not found: {numbered_name}")
    
    # Check if we need to restore the original data.csv
    if not os.path.exists('data.csv') and os.path.exists('sc-data.csv'):
        print("\n📁 Restoring original data.csv...")
        os.rename('sc-data.csv', 'data.csv')
        print("   ✓ Restored: sc-data.csv → data.csv")
    
    print("\n✅ Reset complete! Folder restored to original state.")
    print("\n📋 Current files:")
    
    # Show current directory contents
    current_files = [f for f in os.listdir('.') if f.endswith(('.py', '.csv', '.db'))]
    for file in sorted(current_files):
        print(f"   - {file}")

if __name__ == "__main__":
    # Ask for confirmation before resetting
    print("⚠️  WARNING: This will remove all generated files and reset script names!")
    print("This action cannot be undone.")
    
    response = input("\nAre you sure you want to reset? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        reset_to_original()
    else:
        print("Reset cancelled.")
