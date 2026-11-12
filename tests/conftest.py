"""Pytest configuration for the agent's unit tests.

The scripts listed in `collect_ignore` are manual, interactive checks that need a webcam or a running Ollama server
(run them directly, e.g. `python tests/camera_test.py`). Their function names start with `test_`, so pytest used to
collect them: two errored on missing fixtures and one made a live HTTP call to localhost:11434.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

collect_ignore = [
    "camera_test.py",       # opens webcams
    "vision_test.py",       # Ollama vision model on an image file
    "test_vision_direct.py",  # webcam + Ollama
    "test_ollama.py",       # live Ollama API call
    "test_autonomy.py",     # runs the whole agent for minutes (needs Ollama)
    "find_timeouts.py",     # code search utility
]
