#!/usr/bin/env python3
"""
Camera Testing Utility
This script detects available cameras and tests their functionality.
"""

import cv2
import time
import argparse
import sys
import os
import threading
from typing import List, Dict, Any

def detect_cameras(max_cameras: int = 10) -> List[Dict[str, Any]]:
    """
    Detect available camera devices.
    
    Args:
        max_cameras: Maximum number of camera indices to check
        
    Returns:
        List of available cameras with their properties
    """
    print(f"Detecting cameras (checking indices 0-{max_cameras-1})...")
    
    available_cameras = []
    
    for idx in range(max_cameras):
        print(f"Testing camera index {idx}...", end=" ", flush=True)
        
        try:
            # Try to open the camera
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret:
                    # Camera is usable
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"✓ Available! ({width}x{height} @ {fps:.1f} FPS)")
                    
                    # Get camera information if available
                    camera_info = {
                        'index': idx,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'frame_available': True
                    }
                    
                    # Try to get camera name (backend-specific)
                    try:
                        backend = int(cap.get(cv2.CAP_PROP_BACKEND))
                        camera_info['backend'] = backend
                    except:
                        pass
                    
                    available_cameras.append(camera_info)
                    
                else:
                    print("✗ Opened but couldn't read frame")
                
                # Release the camera
                cap.release()
            else:
                print("✗ Failed to open")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    return available_cameras

def test_camera(camera_idx: int, display: bool = True, test_time: int = 5) -> bool:
    """
    Test a specific camera by displaying a live feed.
    
    Args:
        camera_idx: Camera index to test
        display: Whether to display the camera feed
        test_time: How long to run the test in seconds
        
    Returns:
        Success flag
    """
    try:
        print(f"\nTesting camera {camera_idx} for {test_time} seconds...")
        
        # Open the camera
        cap = cv2.VideoCapture(camera_idx)
        
        if not cap.isOpened():
            print(f"Failed to open camera {camera_idx}")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera opened successfully: {width}x{height} @ {fps:.1f} FPS")
        
        # Create display window if needed
        if display:
            cv2.namedWindow(f"Camera {camera_idx} Test", cv2.WINDOW_NORMAL)
        
        # Track frame count and timing
        start_time = time.time()
        frame_count = 0
        last_update = start_time
        
        # Capture loop
        while (time.time() - start_time) < test_time:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Display the frame
            if display:
                # Add overlay with camera info
                cv2.putText(
                    frame,
                    f"Camera {camera_idx} - {width}x{height}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add frame counter
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Show the frame
                cv2.imshow(f"Camera {camera_idx} Test", frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print statistics every second
            current_time = time.time()
            if current_time - last_update >= 1.0:
                elapsed = current_time - last_update
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Current FPS: {current_fps:.1f}")
                last_update = current_time
                frame_count = 0
        
        # Clean up
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"Camera {camera_idx} test completed successfully")
        return True
    
    except Exception as e:
        print(f"Error testing camera {camera_idx}: {e}")
        return False

def save_camera_frame(camera_idx: int, output_path: str) -> bool:
    """
    Capture a single frame from a camera and save it to a file.
    
    Args:
        camera_idx: Camera index
        output_path: Where to save the frame
        
    Returns:
        Success flag
    """
    try:
        print(f"Capturing frame from camera {camera_idx}...")
        
        # Open the camera
        cap = cv2.VideoCapture(camera_idx)
        
        if not cap.isOpened():
            print(f"Failed to open camera {camera_idx}")
            return False
        
        # Read a few frames to let the camera adjust
        for _ in range(5):
            cap.read()
            time.sleep(0.1)
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame")
            cap.release()
            return False
        
        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"Frame saved to {output_path}")
        
        # Clean up
        cap.release()
        return True
        
    except Exception as e:
        print(f"Error saving camera frame: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Camera Testing Utility")
    parser.add_argument(
        "-d", "--detect",
        action="store_true",
        help="Detect available cameras"
    )
    parser.add_argument(
        "-t", "--test",
        type=int,
        default=None,
        help="Test a specific camera index"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run test without displaying the camera feed"
    )
    parser.add_argument(
        "-c", "--capture",
        type=int,
        default=None,
        help="Capture a single frame from the specified camera index"
    )
    parser.add_argument(
        "-o", "--output",
        default="camera_frame.jpg",
        help="Output path for captured frame"
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=10,
        help="Maximum number of camera indices to check during detection"
    )
    
    args = parser.parse_args()
    
    # Default to detection if no specific action
    if not args.detect and args.test is None and args.capture is None:
        args.detect = True
    
    # Detect cameras
    if args.detect:
        cameras = detect_cameras(args.max_cameras)
        
        if cameras:
            print(f"\n{len(cameras)} camera(s) detected:")
            for cam in cameras:
                print(f"  Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS")
            
            # Suggest next steps
            print("\nTo test a specific camera, run with -t/--test <camera_index>")
            print("To capture a frame from a camera, run with -c/--capture <camera_index>")
        else:
            print("\nNo cameras detected.")
            print("If you believe this is an error, try running with elevated permissions.")
    
    # Test a specific camera
    if args.test is not None:
        display = not args.no_display
        test_camera(args.test, display=display)
    
    # Capture a frame
    if args.capture is not None:
        save_camera_frame(args.capture, args.output)

if __name__ == "__main__":
    main()
