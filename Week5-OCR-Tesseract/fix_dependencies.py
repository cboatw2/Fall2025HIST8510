#!/usr/bin/env python3
"""
Fix Dependencies Script
======================

This script helps resolve common dependency issues with the OCR demo,
particularly the numpy/pandas compatibility issue.

Usage:
    python fix_dependencies.py
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command and show the result."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
        else:
            print("❌ Failed!")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main function to fix dependencies."""
    print("=" * 60)
    print("DEPENDENCY FIX SCRIPT")
    print("=" * 60)
    
    print("This script will help resolve common dependency issues.")
    print("The main issue is usually numpy/pandas compatibility.")
    
    # Commands to try in order
    fix_commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install --upgrade numpy", "Upgrading numpy"),
        ("pip install --upgrade pandas", "Upgrading pandas"),
        ("pip install --upgrade pytesseract", "Upgrading pytesseract"),
        ("pip install --force-reinstall pytesseract", "Force reinstalling pytesseract"),
    ]
    
    print("\nAttempting to fix dependencies...")
    
    for command, description in fix_commands:
        success = run_command(command, description)
        if not success:
            print(f"⚠️  {description} failed, continuing...")
    
    print("\n" + "=" * 60)
    print("ALTERNATIVE SOLUTIONS")
    print("=" * 60)
    
    print("""
If the above didn't work, try these alternatives:

1. CREATE A NEW VIRTUAL ENVIRONMENT:
   python -m venv ocr_env
   source ocr_env/bin/activate  # On Windows: ocr_env\\Scripts\\activate
   pip install -r requirements.txt

2. USE CONDA INSTEAD OF PIP:
   conda install numpy pandas pytesseract
   conda install -c conda-forge opencv matplotlib pillow pdf2image

3. INSTALL SPECIFIC VERSIONS:
   pip install numpy==1.24.3 pandas==2.0.3 pytesseract==0.3.10

4. SKIP OCR DEMO FOR NOW:
   You can still run demos 1-4 (PDF conversion and preprocessing)
   python run_demos.py --demo 1,2,3,4

5. USE SYSTEM TESSERACT:
   Install Tesseract system-wide and use subprocess calls instead
""")
    
    print("\n" + "=" * 60)
    print("TESTING OCR DEMO")
    print("=" * 60)
    
    # Test if OCR demo works now
    test_command = "python ocr_demo.py --check-deps"
    print("Testing OCR demo...")
    
    try:
        result = subprocess.run([sys.executable, "ocr_demo.py", "--check-deps"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ OCR demo dependencies are working!")
            print("You can now run: python ocr_demo.py")
        else:
            print("❌ OCR demo still has issues")
            print("Try the alternative solutions above")
    except Exception as e:
        print(f"❌ Error testing OCR demo: {str(e)}")

if __name__ == "__main__":
    main()
