#!/usr/bin/env python3
"""
Setup Script for Word Vector Alignment Lesson

This script helps set up the environment for the word vector alignment lesson.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    print("-" * 40)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_structure():
    """Check if data directories exist."""
    print("\n📁 Checking data structure...")
    print("-" * 40)
    
    txt_dir = Path("txt")
    period_1900_20 = txt_dir / "1900-20"
    period_1940_60 = txt_dir / "1940-60"
    
    if not txt_dir.exists():
        print("❌ 'txt' directory not found")
        return False
    
    if not period_1900_20.exists():
        print("❌ 'txt/1900-20' directory not found")
        return False
    else:
        txt_files = list(period_1900_20.glob("*.txt"))
        print(f"✅ Found {len(txt_files)} files in 1900-20 period")
    
    if not period_1940_60.exists():
        print("❌ 'txt/1940-60' directory not found")
        return False
    else:
        txt_files = list(period_1940_60.glob("*.txt"))
        print(f"✅ Found {len(txt_files)} files in 1940-60 period")
    
    return True

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing imports...")
    print("-" * 40)
    
    required_modules = [
        "gensim",
        "sklearn",
        "scipy",
        "numpy",
        "pandas"
    ]
    
    all_imported = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_imported = False
    
    return all_imported

def main():
    """Main setup function."""
    print("Word Vector Alignment Lesson Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check data structure
    if not check_data_structure():
        print("\n❌ Data structure check failed. Please ensure:")
        print("   - 'txt/1900-20/' directory exists with .txt files")
        print("   - 'txt/1940-60/' directory exists with .txt files")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Package installation failed. Please install manually:")
        print("   pip install -r requirements.txt")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your installation.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python train_word2vec_models.py")
    print("2. Run: python align_word_vectors.py")
    print("3. Run: python demo_results.py")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()
