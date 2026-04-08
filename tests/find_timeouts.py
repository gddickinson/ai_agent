#!/usr/bin/env python3
"""
Script to find all hardcoded timeouts in the codebase.
"""

import os
import re
import sys
from pathlib import Path

def find_timeouts_in_file(filepath):
    """Find timeout patterns in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Patterns to search for
        patterns = [
            r'timeout\s*=\s*(\d+\.?\d*)',  # timeout = 10.0
            r'timeout\s*:\s*(\d+\.?\d*)',  # timeout: 10.0
            r'timeout[\'"]?\s*:\s*(\d+\.?\d*)',  # "timeout": 10.0
            r'timeout_seconds\s*=\s*(\d+\.?\d*)',  # timeout_seconds = 10.0
            r'\.timeout\((\d+\.?\d*)\)',  # .timeout(10.0)
        ]
        
        matches = []
        line_num = 0
        
        for line in content.split('\n'):
            line_num += 1
            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    timeout_value = match.group(1)
                    matches.append((line_num, line.strip(), timeout_value))
        
        return matches
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

def find_all_timeouts(directory='.'):
    """Find all timeout patterns in Python files in a directory and subdirectories."""
    results = {}
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Skip any "hidden" directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        # Process Python files
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                timeouts = find_timeouts_in_file(path)
                if timeouts:
                    results[path] = timeouts
    
    return results

def main():
    print("Searching for hardcoded timeouts...")
    
    # Find all timeout patterns
    timeouts = find_all_timeouts()
    
    if not timeouts:
        print("No hardcoded timeouts found.")
        return 0
    
    # Print results
    print(f"Found {sum(len(v) for v in timeouts.values())} hardcoded timeouts in {len(timeouts)} files:")
    print()
    
    for filepath, matches in sorted(timeouts.items()):
        print(f"\n{filepath}:")
        for line_num, line, timeout_value in matches:
            print(f"  Line {line_num}: {line} (timeout: {timeout_value}s)")
    
    # Files of interest
    print("\nFiles of special interest:")
    vision_files = [f for f in timeouts if "vision" in f.lower() or "perception" in f.lower() or "manager" in f.lower()]
    for f in vision_files:
        print(f"- {f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
