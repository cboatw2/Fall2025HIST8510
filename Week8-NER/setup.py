#!/usr/bin/env python3
"""
Setup script for NER analysis environment.
Installs required packages and downloads spaCy model.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up NER Analysis Environment")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected. Consider creating one:")
        print("   python3 -m venv ner_env")
        print("   source ner_env/bin/activate")
        print()
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install packages. Please check your Python environment.")
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("❌ Failed to download spaCy model. Please try manually:")
        print("   python -m spacy download en_core_web_sm")
        return False
    
    print("\n🎉 Setup complete! You can now run:")
    print("   python ner_analysis.py APER_1942_February.txt")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

