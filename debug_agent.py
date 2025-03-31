#!/usr/bin/env python3
"""
Simple Debug Tool for Embodied AI Agent
Tests basic functionality of Ollama and messaging.
"""

import sys
import os
import time
import logging
import json
import requests
import traceback
from pathlib import Path

def test_ollama():
    """Test basic connection to Ollama API."""
    print("\n=== Testing Ollama API Connection ===")

    api_base = "http://localhost:11434/api"

    try:
        # Test API connection
        print("Checking if Ollama service is reachable...")
        response = requests.get(f"{api_base}/tags", timeout=5.0)

        if response.status_code != 200:
            print(f"Error: Could not connect to Ollama API. Status code: {response.status_code}")
            return False

        print("✓ Ollama API is reachable")

        # Check available models
        models = response.json().get('models', [])
        model_names = [model.get('name') for model in models]

        print(f"Available models: {model_names}")

        # Test a simple generation
        print("\nTesting simple generation with first available model...")
        model_to_test = model_names[0] if model_names else "llama3:latest"

        payload = {
            "model": model_to_test,
            "prompt": "Say hello in exactly one word.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 10
            }
        }

        start_time = time.time()
        response = requests.post(f"{api_base}/generate", json=payload, timeout=10.0)
        end_time = time.time()

        if response.status_code != 200:
            print(f"Error: Generation failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        response_text = result.get('response', '')

        print(f"✓ Generation successful in {end_time - start_time:.2f} seconds")
        print(f"Response: {response_text}")

        return True

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API. Is Ollama running?")
        return False
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_json_generation():
    """Test if Ollama can generate proper JSON responses."""
    print("\n=== Testing JSON Generation ===")

    api_base = "http://localhost:11434/api"
    model_id = "llama3:latest"

    prompt = """
    You are an AI assistant. Respond to the following message from a human:

    Message: "Hello! Can you hear me?"

    Your response should be in JSON format with these fields:
    {
      "content": "Your actual response to the human",
      "thought": "Your internal thought about this interaction",
      "emotion": "The primary emotion you're expressing"
    }
    """

    try:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512
            }
        }

        print(f"Testing generation with {model_id}...")

        response = requests.post(f"{api_base}/generate", json=payload, timeout=20.0)
        if response.status_code != 200:
            print(f"Error: Generation failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        response_text = result.get('response', '')

        print(f"Raw response: {response_text[:150]}...")

        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx+1]
            try:
                parsed = json.loads(json_str)
                print(f"\nJSON parsed successfully: {json.dumps(parsed, indent=2)[:200]}...")
                return True
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return False
        else:
            print("Could not find JSON in response")
            return False

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_direct_output():
    """Test writing directly to the output files."""
    print("\n=== Testing Direct Output ===")

    try:
        output_dir = Path("memory/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        display_file = output_dir / "display.txt"
        speech_file = output_dir / "speech.txt"

        # Test display output
        with open(display_file, 'a') as f:
            f.write(f"\n[{time.ctime()}] TEST: This is a direct test message to the display output.\n")

        # Test speech output
        with open(speech_file, 'a') as f:
            f.write(f"\n[{time.ctime()}] TEST SPEECH: This is a direct test message to the speech output.\n")

        print(f"✓ Test messages written to output files")
        print(f"  Display: {display_file}")
        print(f"  Speech: {speech_file}")

        return True

    except Exception as e:
        print(f"Error writing to output files: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Simple Debug Tool for Embodied AI Agent ===")

    # Run tests
    ollama_ok = test_ollama()
    json_ok = test_json_generation() if ollama_ok else False
    output_ok = test_direct_output()

    # Print summary
    print("\n=== Summary ===")
    print(f"Ollama Connection: {'✓' if ollama_ok else '✗'}")
    print(f"JSON Generation: {'✓' if json_ok else '✗'}")
    print(f"Direct Output: {'✓' if output_ok else '✗'}")

    # Recommendations
    print("\n=== Recommendations ===")
    if not ollama_ok:
        print("- Make sure Ollama is running")
        print("- Verify the model 'llama3:latest' is available (run: ollama pull llama3)")

    if not json_ok:
        print("- The LLM is having trouble generating valid JSON")
        print("- Try increasing the token limit in your prompt")

    if not output_ok:
        print("- Check permissions for the memory/output directory")

    print("\nTry the enhanced_chat.py next for a more reliable interface")
