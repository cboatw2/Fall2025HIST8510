#!/usr/bin/env python3
"""
Project Reset Script for Word Vector Alignment Lesson

This script cleans up generated files while preserving the core models and source data.
It removes:
- All visualization plots (semantic_shift_plots/, aligned_semantic_plots/, comparison_plots/)
- Alignment results (alignment_results/)
- Any cached/temporary files

It preserves:
- Word2Vec models (*.model, *.pkl)
- Source text files (txt/ directory)
- Python scripts
- Requirements and documentation

Author: Lesson Plan for Digital Humanities Course
"""

import os
import shutil
import glob
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_project():
    """Reset the project by removing generated files while preserving core data."""
    
    logger.info("🧹 Starting project reset...")
    logger.info("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    logger.info(f"Project root: {project_root}")
    
    # Define directories and files to remove
    directories_to_remove = [
        "semantic_shift_plots",
        "aligned_semantic_plots", 
        "comparison_plots",
        "alignment_results",
        "__pycache__"
    ]
    
    # Define file patterns to remove
    file_patterns_to_remove = [
        "*.pyc",
        "*.pyo", 
        "*.pyd",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    # Remove directories
    removed_dirs = []
    for dir_name in directories_to_remove:
        dir_path = project_root / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_name)
                logger.info(f"✅ Removed directory: {dir_name}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {dir_name}: {e}")
        else:
            logger.info(f"ℹ️  Directory not found: {dir_name}")
    
    # Remove files matching patterns
    removed_files = []
    for pattern in file_patterns_to_remove:
        matches = list(project_root.glob(pattern))
        matches.extend(list(project_root.glob(f"**/{pattern}")))
        
        for file_path in matches:
            try:
                file_path.unlink()
                removed_files.append(str(file_path.relative_to(project_root)))
                logger.info(f"✅ Removed file: {file_path.relative_to(project_root)}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {file_path}: {e}")
    
    # Check what's preserved
    logger.info("=" * 50)
    logger.info("📁 PRESERVED FILES AND DIRECTORIES:")
    logger.info("=" * 50)
    
    preserved_items = []
    
    # Check for Word2Vec models
    model_files = list(project_root.glob("*.model")) + list(project_root.glob("*.pkl"))
    if model_files:
        logger.info("🤖 Word2Vec Models:")
        for model_file in model_files:
            logger.info(f"   ✅ {model_file.name}")
            preserved_items.append(model_file.name)
    
    # Check for source text files
    txt_dir = project_root / "txt"
    if txt_dir.exists():
        txt_count = len(list(txt_dir.rglob("*.txt")))
        logger.info(f"📚 Source Text Files: {txt_count} files in txt/ directory")
        preserved_items.append("txt/ directory")
    
    # Check for Python scripts
    python_files = list(project_root.glob("*.py"))
    logger.info("🐍 Python Scripts:")
    for py_file in python_files:
        logger.info(f"   ✅ {py_file.name}")
        preserved_items.append(py_file.name)
    
    # Check for documentation
    doc_files = list(project_root.glob("*.md")) + list(project_root.glob("*.txt"))
    if doc_files:
        logger.info("📖 Documentation:")
        for doc_file in doc_files:
            logger.info(f"   ✅ {doc_file.name}")
            preserved_items.append(doc_file.name)
    
    # Check for requirements
    req_files = list(project_root.glob("requirements*.txt"))
    if req_files:
        logger.info("📦 Requirements:")
        for req_file in req_files:
            logger.info(f"   ✅ {req_file.name}")
            preserved_items.append(req_file.name)
    
    # Check for virtual environment
    venv_dirs = [d for d in project_root.iterdir() if d.is_dir() and "env" in d.name.lower()]
    if venv_dirs:
        logger.info("🔧 Virtual Environment:")
        for venv_dir in venv_dirs:
            logger.info(f"   ✅ {venv_dir.name}")
            preserved_items.append(venv_dir.name)
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 RESET SUMMARY:")
    logger.info("=" * 50)
    logger.info(f"🗑️  Removed directories: {len(removed_dirs)}")
    for dir_name in removed_dirs:
        logger.info(f"   - {dir_name}")
    
    logger.info(f"🗑️  Removed files: {len(removed_files)}")
    for file_name in removed_files[:10]:  # Show first 10
        logger.info(f"   - {file_name}")
    if len(removed_files) > 10:
        logger.info(f"   ... and {len(removed_files) - 10} more files")
    
    logger.info(f"✅ Preserved items: {len(preserved_items)}")
    
    logger.info("=" * 50)
    logger.info("🎯 NEXT STEPS FOR STUDENTS:")
    logger.info("=" * 50)
    logger.info("1. Activate virtual environment:")
    logger.info("   source word_vector_env/bin/activate")
    logger.info("")
    logger.info("2. Run word vector alignment:")
    logger.info("   python align_word_vectors.py")
    logger.info("")
    logger.info("3. Create visualizations:")
    logger.info("   python visualize_aligned_shifts.py <word> --context-words 15")
    logger.info("   python compare_aligned_unaligned.py <word> --context-words 15")
    logger.info("")
    logger.info("4. Explore interactively:")
    logger.info("   python interactive_semantic_shifts.py")
    logger.info("")
    logger.info("✅ Project reset complete! Ready for student use.")

def main():
    """Main function with confirmation prompt."""
    import sys
    
    print("🧹 Word Vector Alignment Project Reset")
    print("=" * 50)
    print("This will remove:")
    print("  - All visualization plots")
    print("  - Alignment results")
    print("  - Cached/temporary files")
    print("")
    print("This will preserve:")
    print("  - Word2Vec models (*.model, *.pkl)")
    print("  - Source text files (txt/ directory)")
    print("  - Python scripts")
    print("  - Documentation and requirements")
    print("")
    
    # Check for command line argument to skip confirmation
    if len(sys.argv) > 1 and sys.argv[1] == "--yes":
        print("Auto-confirming reset...")
        reset_project()
        return
    
    try:
        response = input("Continue with reset? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            reset_project()
        else:
            print("❌ Reset cancelled.")
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Reset cancelled.")

if __name__ == "__main__":
    main()
