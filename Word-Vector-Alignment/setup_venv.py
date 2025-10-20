#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Word Vector Alignment

This script creates a virtual environment and installs all required packages
for the word vector alignment lesson.
"""

import subprocess
import sys
import os
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment in the current directory."""
    print("Creating virtual environment...")
    
    venv_name = "word_vector_env"
    
    try:
        # Try to use Python 3.11 if available, otherwise use current Python
        python_executable = "python3.11"
        try:
            subprocess.check_call([python_executable, "--version"], capture_output=True)
            print(f"Using {python_executable} for virtual environment")
        except (subprocess.CalledProcessError, FileNotFoundError):
            python_executable = sys.executable
            print(f"Python 3.11 not found, using {python_executable}")
        
        # Create virtual environment
        subprocess.check_call([python_executable, "-m", "venv", venv_name])
        print(f"✅ Virtual environment '{venv_name}' created successfully!")
        return venv_name
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return None

def get_activation_commands(venv_name):
    """Get the commands needed to activate the virtual environment."""
    if os.name == 'nt':  # Windows
        activate_script = Path(venv_name) / "Scripts" / "activate.bat"
        return f"{activate_script}"
    else:  # Unix/Linux/macOS
        activate_script = Path(venv_name) / "bin" / "activate"
        return f"source {activate_script}"

def install_requirements(venv_name):
    """Install requirements in the virtual environment."""
    print(f"\nInstalling requirements in virtual environment '{venv_name}'...")
    
    # Determine the pip executable path
    if os.name == 'nt':  # Windows
        pip_path = Path(venv_name) / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        pip_path = Path(venv_name) / "bin" / "pip"
    
    try:
        # Install requirements
        subprocess.check_call([str(pip_path), "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_activation_script(venv_name):
    """Create a convenient activation script."""
    if os.name == 'nt':  # Windows
        script_content = f"""@echo off
echo Activating Word Vector Alignment virtual environment...
call {venv_name}\\Scripts\\activate.bat
echo Virtual environment activated!
echo.
echo You can now run:
echo   python train_word2vec_models.py
echo   python align_word_vectors.py
echo   python demo_results.py
echo.
echo To deactivate, type: deactivate
"""
        script_path = "activate_env.bat"
    else:  # Unix/Linux/macOS
        script_content = f"""#!/bin/bash
echo "Activating Word Vector Alignment virtual environment..."
source {venv_name}/bin/activate
echo "Virtual environment activated!"
echo ""
echo "You can now run:"
echo "  python train_word2vec_models.py"
echo "  python align_word_vectors.py"
echo "  python demo_results.py"
echo ""
echo "To deactivate, type: deactivate"
"""
        script_path = "activate_env.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"✅ Created activation script: {script_path}")

def main():
    """Main setup function."""
    print("Word Vector Alignment - Virtual Environment Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found in current directory")
        print("Please run this script from the Word-Vector-Alignment directory")
        return
    
    # Create virtual environment
    venv_name = create_virtual_environment()
    if not venv_name:
        return
    
    # Install requirements
    if not install_requirements(venv_name):
        print("❌ Failed to install requirements")
        return
    
    # Create activation script
    create_activation_script(venv_name)
    
    print("\n🎉 Virtual environment setup completed!")
    print("\nTo use the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("1. Run: activate_env.bat")
        print("   OR")
        print(f"2. Run: {venv_name}\\Scripts\\activate.bat")
    else:  # Unix/Linux/macOS
        print("1. Run: source activate_env.sh")
        print("   OR")
        print(f"2. Run: source {venv_name}/bin/activate")
    
    print("\nThen you can run the lesson scripts:")
    print("  python train_word2vec_models.py")
    print("  python align_word_vectors.py")
    print("  python demo_results.py")
    
    print(f"\nTo deactivate the virtual environment, type: deactivate")

if __name__ == "__main__":
    main()
