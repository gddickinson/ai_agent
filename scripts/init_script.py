#!/usr/bin/env python3
"""
Project Initialization Script
Sets up the directory structure and initial files for the Embodied AI Agent project.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    print("Creating directory structure...")
    
    # Define directories to create
    directories = [
        "core",
        "hardware",
        "hardware/adapters",
        "llm",
        "utils",
        "memory",
        "memory/episodic",
        "memory/semantic",
        "memory/output",
        "logs",
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_dirs = [
        "core",
        "hardware",
        "hardware/adapters",
        "llm",
        "utils",
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"Created file: {init_file}")

def create_empty_files():
    """Create empty files for modules that don't exist yet."""
    # Define files to create if they don't exist
    files = [
        "hardware/adapters/mac_adapters.py",
        "hardware/adapters/robot_adapters.py",
    ]
    
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            # Create directory if it doesn't exist
            path.parent.mkdir(exist_ok=True, parents=True)
            
            # Create file with basic content
            with open(path, 'w') as f:
                f.write(f'"""\n{path.stem} Module\n"""\n\n')
            
            print(f"Created file: {file_path}")

def print_next_steps():
    """Print next steps to take after initialization."""
    print("\nProject initialized successfully!")
    print("\nNext steps:")
    print("1. Install required packages:")
    print("   pip install pyyaml requests")
    print("2. Make sure Ollama is installed and running:")
    print("   https://ollama.ai/")
    print("3. Pull the llama3 model:")
    print("   ollama pull llama3")
    print("4. Run the agent:")
    print("   python main.py")
    print("\nHappy hacking!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Initialize Embodied AI Agent project")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    args = parser.parse_args()
    
    create_directory_structure()
    create_empty_files()
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
