#!/usr/bin/env python3
"""
Monitors the agent's output files in real-time.
"""

import os
import sys
import time
import argparse
from pathlib import Path

def monitor_file(file_path, label):
    """Monitor a file for changes and print new content."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Get initial file size
    current_size = os.path.getsize(file_path)
    
    # Open file for reading
    with open(file_path, 'r') as f:
        # Seek to the end
        f.seek(current_size)
        
        while True:
            # Read any new content
            line = f.readline()
            if line:
                print(f"[{label}] {line}", end='')
                sys.stdout.flush()
            else:
                # Wait for more content
                time.sleep(0.1)
                
                # Check if file size has changed
                new_size = os.path.getsize(file_path)
                if new_size < current_size:
                    # File was truncated, seek to beginning
                    f.seek(0)
                    current_size = 0
                else:
                    current_size = new_size

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor agent output files")
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Monitor display output"
    )
    parser.add_argument(
        "--speech", 
        action="store_true", 
        help="Monitor speech output"
    )
    parser.add_argument(
        "--log", 
        action="store_true", 
        help="Monitor log file"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Monitor all output files"
    )
    parser.add_argument(
        "--log-file", 
        default=None,
        help="Path to specific log file to monitor"
    )
    args = parser.parse_args()
    
    # If no specific flags, default to --all
    if not (args.display or args.speech or args.log or args.log_file):
        args.all = True
    
    # Define file paths
    memory_dir = Path("memory")
    display_file = memory_dir / "output" / "display.txt"
    speech_file = memory_dir / "output" / "speech.txt"
    
    # Find most recent log file if not specified
    if args.log or args.all:
        if args.log_file:
            log_file = Path(args.log_file)
        else:
            log_dir = Path("logs")
            if log_dir.exists():
                log_files = sorted(log_dir.glob("agent_*.log"), key=os.path.getmtime, reverse=True)
                if log_files:
                    log_file = log_files[0]
                else:
                    print("No log files found in logs directory")
                    log_file = None
            else:
                print("Logs directory not found")
                log_file = None
    else:
        log_file = None
    
    # Start monitoring
    try:
        # Display banner
        print("=== Monitoring Agent Output ===")
        print("Press Ctrl+C to stop")
        print("=============================")
        
        # Start monitoring threads
        import threading
        
        threads = []
        
        if (args.display or args.all) and display_file.exists():
            print(f"Monitoring display output: {display_file}")
            display_thread = threading.Thread(
                target=monitor_file,
                args=(display_file, "DISPLAY"),
                daemon=True
            )
            display_thread.start()
            threads.append(display_thread)
        
        if (args.speech or args.all) and speech_file.exists():
            print(f"Monitoring speech output: {speech_file}")
            speech_thread = threading.Thread(
                target=monitor_file,
                args=(speech_file, "SPEECH"),
                daemon=True
            )
            speech_thread.start()
            threads.append(speech_thread)
        
        if log_file and (args.log or args.all):
            print(f"Monitoring log file: {log_file}")
            log_thread = threading.Thread(
                target=monitor_file,
                args=(log_file, "LOG"),
                daemon=True
            )
            log_thread.start()
            threads.append(log_thread)
        
        # Wait for threads
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    
    return 0

if __name__ == "__main__":
    exit(main())
