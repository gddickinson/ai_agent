#!/usr/bin/env python3
"""
Simple test script for Ollama API
"""

import requests
import json
import time

def test_ollama():
    """Test basic Ollama API functionality."""
    print("Testing Ollama API...")

    api_base = "http://localhost:11434/api"
    model_id = "llama3:latest"

    # Test API connection
    try:
        # First check if the service is reachable
        response = requests.get(f"{api_base}/tags")
        if response.status_code != 200:
            print(f"Error: Couldn't connect to Ollama API. Status code: {response.status_code}")
            return False

        print("✓ Ollama API is reachable")

        # Check if the model is available
        models = response.json().get('models', [])
        model_found = any(model.get('name') == model_id for model in models)

        if not model_found:
            print(f"Warning: Model '{model_id}' not found in available models.")
            print(f"Available models: {[model.get('name') for model in models]}")
            return False

        print(f"✓ Model '{model_id}' is available")

        # Test generating a response
        print("Testing generation...")
        start_time = time.time()

        payload = {
            "model": model_id,
            "prompt": "Hello! Please respond with a very short greeting.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 50
            }
        }

        response = requests.post(f"{api_base}/generate", json=payload)

        if response.status_code != 200:
            print(f"Error: Generation failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        response_text = result.get('response', '')

        end_time = time.time()
        duration = end_time - start_time

        print(f"✓ Generation successful in {duration:.2f} seconds")
        print(f"Response: {response_text}")

        # Print some more information about the response
        print(f"Prompt eval count: {result.get('prompt_eval_count', 0)}")
        print(f"Eval count: {result.get('eval_count', 0)}")
        print(f"Total tokens: {result.get('prompt_eval_count', 0) + result.get('eval_count', 0)}")

        return True

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API. Is Ollama running?")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_ollama()
