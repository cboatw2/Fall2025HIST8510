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
    
    # Find all CSV files
    csv_files = glob.glob("*.csv")
    
    if csv_files:
        print(f"\n🗑️  Found {len(csv_files)} CSV files to remove:")
        for csv_file in csv_files:
            print(f"   • {csv_file}")
        
        # Remove CSV files
        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                print(f"✅ Removed: {csv_file}")
            except OSError as e:
                print(f"❌ Error removing {csv_file}: {e}")
    else:
        print("\n📄 No CSV files found to remove")
    
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
