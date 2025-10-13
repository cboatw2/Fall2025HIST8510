#!/usr/bin/env python3
"""
Reset script for Week8-NER directory.
Removes all CSV files and cleans up the directory.
"""

import os
import glob
import shutil
from pathlib import Path

def reset_directory():
    """Reset the Week8-NER directory by removing CSV files and cleaning up."""
    print("🧹 Resetting Week8-NER directory...")
    print("=" * 50)
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Find all CSV files in current directory and results directory
    csv_files_current = glob.glob("*.csv")
    csv_files_results = glob.glob("results/*.csv") if os.path.exists("results") else []
    all_csv_files = csv_files_current + csv_files_results
    
    if all_csv_files:
        print(f"\n🗑️  Found {len(all_csv_files)} CSV files to remove:")
        for csv_file in all_csv_files:
            print(f"   • {csv_file}")
        
        # Remove CSV files
        for csv_file in all_csv_files:
            try:
                os.remove(csv_file)
                print(f"✅ Removed: {csv_file}")
            except OSError as e:
                print(f"❌ Error removing {csv_file}: {e}")
    else:
        print("\n📄 No CSV files found to remove")
    
    # Remove results directory if it exists and is empty
    results_dir = Path("results")
    if results_dir.exists():
        try:
            if not any(results_dir.iterdir()):  # Directory is empty
                results_dir.rmdir()
                print("✅ Removed empty results directory")
            else:
                print("ℹ️  Results directory not empty, keeping it")
        except OSError as e:
            print(f"ℹ️  Could not remove results directory: {e}")
    
    # Check for virtual environment
    venv_dir = Path("ner_env")
    if venv_dir.exists():
        print(f"\n🐍 Virtual environment found: {venv_dir}")
        try:
            response = input("   Do you want to remove the virtual environment? (y/N): ")
            if response.lower() in ['y', 'yes']:
                shutil.rmtree(venv_dir)
                print("✅ Removed virtual environment")
            else:
                print("ℹ️  Virtual environment kept")
        except EOFError:
            print("ℹ️  Virtual environment kept (non-interactive mode)")
    
    # List remaining files
    remaining_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    if remaining_files:
        print(f"\n📋 Remaining files in directory:")
        for file in sorted(remaining_files):
            print(f"   • {file}")
    else:
        print(f"\n📋 No files remaining in directory")
    
    print(f"\n🎉 Reset complete!")
    print(f"   To set up again, run: python setup.py")

def main():
    """Main function."""
    try:
        reset_directory()
    except KeyboardInterrupt:
        print("\n\n⚠️  Reset cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during reset: {e}")

if __name__ == "__main__":
    main()
