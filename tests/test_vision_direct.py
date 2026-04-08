#!/usr/bin/env python3
"""
Direct Vision Test
This script tests the vision model directly by taking a webcam snapshot
and sending it to the Ollama vision model.
"""

import cv2
import base64
import requests
import json
import time
import argparse
import sys

def capture_camera_frame(camera_idx=1, width=640, height=480):
    """Capture a frame from the camera."""
    print(f"Attempting to access camera {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Capture a frame
    ret, frame = cap.read()
    
    # Release the camera
    cap.release()
    
    if not ret:
        print("Error: Failed to capture frame")
        return None
    
    print(f"Successfully captured frame with shape {frame.shape}")
    return frame

def process_image_with_ollama(frame, model_name="llama3.2-vision:latest", timeout=200):
    """Process the image with Ollama vision model."""
    if frame is None:
        print("Error: No frame to process")
        return None
    
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare request payload
    prompt = "Please describe this image in detail. What do you see?"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 256
        }
    }
    
    print(f"Sending image to Ollama vision model (timeout: {timeout}s)...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        print(f"Received response in {elapsed_time:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response provided")
        else:
            print(f"Error: {response.status_code} {response.reason}")
            print(response.text)
            return None
    
    except requests.exceptions.Timeout:
        elapsed_time = time.time() - start_time
        print(f"Error: Request timed out after {elapsed_time:.1f} seconds")
        return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_image(frame, filename="test_image.jpg"):
    """Save the image to a file."""
    if frame is None:
        print("Error: No frame to save")
        return False
    
    try:
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Ollama Vision Model with Webcam")
    parser.add_argument(
        "--camera", 
        type=int,
        default=1,
        help="Camera index (default: 1)"
    )
    parser.add_argument(
        "--model",
        default="llama3.2-vision:latest",
        help="Ollama vision model name"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=200.0,
        help="Timeout in seconds (default: 200)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the captured image"
    )
    
    args = parser.parse_args()
    
    # Capture frame
    frame = capture_camera_frame(args.camera)
    if frame is None:
        return 1
    
    # Save image if requested
    if args.save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_image(frame, f"snapshot_{timestamp}.jpg")
    
    # Process with vision model
    description = process_image_with_ollama(
        frame, 
        model_name=args.model,
        timeout=args.timeout
    )
    
    if description:
        print("\nImage Description:")
        print("="*50)
        print(description)
        print("="*50)
        return 0
    else:
        print("Failed to get description from vision model")
        return 1

if __name__ == "__main__":
    sys.exit(main())
