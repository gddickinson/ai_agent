#!/usr/bin/env python3
"""
Test script for Ollama vision model connectivity
This tests the direct API connection to Ollama for vision models
"""

import requests
import argparse
import base64
import json
import sys
from pathlib import Path
import time

def test_ollama_vision(image_path, model_name="llama3.2-vision:latest"):
    """
    Test Ollama vision model with a local image file.

    Args:
        image_path: Path to image file
        model_name: Ollama model to use
    """
    print(f"Testing Ollama vision model: {model_name}")
    print(f"Image path: {image_path}")

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return False

    # Read the image file and convert to base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

    print(f"Successfully read image, size: {len(image_data)} bytes")

    # Create payload for Ollama API
    prompt = "Please describe this image in detail. What do you see?"

    # Test both API formats to see which one works

    # Test Format 1: Using the /api/generate endpoint with images array
    payload1 = {
        "model": model_name,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 256
        }
    }

    print("\nTesting Format 1: /api/generate with images array")
    try:
        start_time = time.time()
        response1 = requests.post(
            "http://localhost:11434/api/generate",
            json=payload1,
            timeout=3000.0
        )
        request_time = time.time() - start_time

        print(f"Response status code: {response1.status_code} (in {request_time:.2f}s)")

        if response1.status_code == 200:
            result1 = response1.json()
            print(f"Response text: {result1.get('response', '')[0:200]}...")
            print("Format 1 SUCCESS")
        else:
            print(f"Error response: {response1.text}")
            print("Format 1 FAILED")
    except Exception as e:
        print(f"Exception with Format 1: {e}")
        print("Format 1 FAILED")

    # Test Format 2: Using the /api/chat endpoint
    payload2 = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    print("\nTesting Format 2: /api/chat with image_url")
    try:
        start_time = time.time()
        response2 = requests.post(
            "http://localhost:11434/api/chat",
            json=payload2,
            timeout=30.0
        )
        request_time = time.time() - start_time

        print(f"Response status code: {response2.status_code} (in {request_time:.2f}s)")

        if response2.status_code == 200:
            result2 = response2.json()
            print(f"Response text: {result2.get('message', {}).get('content', '')[0:200]}...")
            print("Format 2 SUCCESS")
        else:
            print(f"Error response: {response2.text}")
            print("Format 2 FAILED")
    except Exception as e:
        print(f"Exception with Format 2: {e}")
        print("Format 2 FAILED")

def main():
    parser = argparse.ArgumentParser(description="Test Ollama Vision Model")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument(
        "--model",
        default="llama3.2-vision:latest",
        help="Ollama vision model name"
    )

    args = parser.parse_args()

    test_ollama_vision(args.image_path, args.model)

    return 0

if __name__ == "__main__":
    sys.exit(main())
