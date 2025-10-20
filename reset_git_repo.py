#!/usr/bin/env python3
"""
Git Repository Reset Script

This script helps clean up uncommitted changes and untracked files
in the git repository while preserving important files.

Author: Lesson Plan for Digital Humanities Course
"""

import os
import subprocess
import sys
from pathlib import Path

def run_git_command(command, description):
    """Run a git command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}")
            if result.stdout.strip():
                print(f"   {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed:")
            print(f"   {result.stderr.strip()}")
        return result
    except Exception as e:
        print(f"❌ Error running '{command}': {e}")
        return None

def reset_git_repository():
    """Reset the git repository to a clean state."""
    
    print("🧹 Git Repository Reset")
    print("=" * 50)
    
    # Check current status
    print("📊 Current Git Status:")
    run_git_command("git status --porcelain", "Checking repository status")
    
    print("\n" + "=" * 50)
    print("🔄 Git Reset Options:")
    print("=" * 50)
    print("1. Reset all changes (modified tracked files)")
    print("2. Remove untracked files")
    print("3. Remove untracked files and directories")
    print("4. Hard reset (discard all changes)")
    print("5. Clean everything (reset + remove untracked)")
    print("6. Show what would be removed (dry run)")
    print("")
    
    try:
        choice = input("Choose an option (1-6): ").strip()
        
        if choice == "1":
            print("\n🔄 Resetting modified tracked files...")
            run_git_command("git reset --hard HEAD", "Reset all tracked files to last commit")
            
        elif choice == "2":
            print("\n🗑️  Removing untracked files...")
            run_git_command("git clean -f", "Remove untracked files")
            
        elif choice == "3":
            print("\n🗑️  Removing untracked files and directories...")
            run_git_command("git clean -fd", "Remove untracked files and directories")
            
        elif choice == "4":
            print("\n⚠️  Hard reset (discard all changes)...")
            confirm = input("This will discard ALL changes. Continue? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_git_command("git reset --hard HEAD", "Hard reset tracked files")
                run_git_command("git clean -fd", "Remove untracked files and directories")
            else:
                print("❌ Hard reset cancelled")
                
        elif choice == "5":
            print("\n🧹 Complete clean (reset + remove untracked)...")
            confirm = input("This will discard ALL changes and remove untracked files. Continue? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_git_command("git reset --hard HEAD", "Reset tracked files")
                run_git_command("git clean -fdx", "Remove all untracked files and directories")
            else:
                print("❌ Complete clean cancelled")
                
        elif choice == "6":
            print("\n🔍 Dry run - showing what would be removed:")
            run_git_command("git clean -fd --dry-run", "Show untracked files that would be removed")
            
        else:
            print("❌ Invalid choice")
            return
            
        print("\n" + "=" * 50)
        print("📊 Final Status:")
        run_git_command("git status", "Final repository status")
        
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Operation cancelled")

def main():
    """Main function."""
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Not in a git repository")
        return
    
    reset_git_repository()

if __name__ == "__main__":
    main()
