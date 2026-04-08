"""
Sensors Module
Handles interfacing with various sensor hardware on different platforms.
"""

import logging
import traceback
import time
import threading
import queue
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import json
import os
import base64
import random
import numpy as np
from PIL import Image
import io
import cv2

class SensorInterface(ABC):
    """Base class for sensor interfaces."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sensor interface.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config
        self.name = config.get('name', 'unknown')
        self.type = config.get('type', 'unknown')
        self.enabled = config.get('enabled', True)
        self.running = False

    @abstractmethod
    def start(self):
        """Start the sensor."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the sensor."""
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        Get sensor data.

        Returns:
            Sensor data dictionary
        """
        pass

class CameraSensor:
    """Interface for camera sensors."""

    def __init__(self, config):
        """Initialize camera sensor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config
        self.name = config.get('name', 'camera')
        self.type = 'camera'
        self.enabled = config.get('enabled', True)
        self.running = False

        # Frame capture settings
        self.frame_interval = config.get('frame_interval', 1.0)  # seconds between frames
        self.last_capture_time = 0
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=3)  # Buffer a few frames

        # Camera device settings
        self.camera_device = config.get('device', 0)  # Default webcam
        self.capture_width = config.get('width', 640)
        self.capture_height = config.get('height', 480)

        # Simulation settings (for testing)
        self.simulate = config.get('simulate', False)
        self.simulate_data = config.get('simulate_data', [
            "A desk with a computer and some papers.",
            "A person sitting at a desk, working on a computer.",
            "An empty room with white walls and a window."
        ])

        # Debug flags
        self.debug_camera_init = config.get('debug_camera_init', False)

        # OpenCV video capture
        self.cap = None
        self.capture_thread = None

        self.logger.info(f"Initialized camera sensor: {self.name}")

    def start(self):
        """Start the camera sensor."""
        if self.running or not self.enabled:
            return

        self.running = True

        if self.simulate:
            # Start simulation thread
            self.logger.info(f"Starting simulated camera: {self.name}")
        else:
            # Initialize the camera
            try:
                self.logger.info(f"Attempting to open camera device: {self.camera_device}")

                # If we're in debug mode, try to detect all available cameras
                if self.debug_camera_init:
                    self._detect_available_cameras()

                # Open the camera
                self.cap = cv2.VideoCapture(self.camera_device)
                if not self.cap.isOpened():
                    self.logger.error(f"Failed to open camera device {self.camera_device}")

                    # Try alternative camera devices if specified
                    alt_devices = self.config.get('alternative_devices', [])
                    for alt_device in alt_devices:
                        self.logger.info(f"Trying alternative camera device: {alt_device}")
                        self.cap = cv2.VideoCapture(alt_device)
                        if self.cap.isOpened():
                            self.logger.info(f"Successfully opened alternative camera device: {alt_device}")
                            self.camera_device = alt_device
                            break

                    # If still not open, fall back to simulation
                    if not self.cap.isOpened():
                        self.logger.warning("All camera devices failed, falling back to simulation mode")
                        self.simulate = True
                        return

                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

                # Get actual resolution (may differ from requested)
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self.logger.info(f"Camera initialized with resolution {actual_width}x{actual_height}")

                # Test capture - try to get a single frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.logger.error("Camera opened but failed to capture test frame")
                    self.simulate = True
                    self.cap.release()
                    self.cap = None
                    return

                # Log successful test capture
                self.logger.info(f"Successfully captured test frame with shape: {test_frame.shape}")

                # Start capture thread
                self.capture_thread = threading.Thread(
                    target=self._capture_loop,
                    name=f"camera_{self.name}_thread",
                    daemon=True
                )
                self.capture_thread.start()
                self.logger.info(f"Started camera capture thread")

            except Exception as e:
                self.logger.error(f"Error initializing camera: {e}")
                self.logger.error(traceback.format_exc())
                self.simulate = True
                self.logger.warning("Falling back to simulation mode due to error")

    def _detect_available_cameras(self):
        """Debug function to detect all available cameras."""
        self.logger.info("Detecting available cameras...")

        # Try up to 3 camera indices
        max_cameras = 3
        available_cameras = []

        for idx in range(max_cameras):
            try:
                # Try to open the camera
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Try to read a frame
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(idx)
                        self.logger.info(f"Camera index {idx} is available")
                    else:
                        self.logger.info(f"Camera index {idx} opened but frame read failed")
                    cap.release()
                else:
                    self.logger.info(f"Camera index {idx} failed to open")
            except Exception as e:
                self.logger.info(f"Error testing camera index {idx}: {e}")

        if available_cameras:
            self.logger.info(f"Available camera indices: {available_cameras}")
        else:
            self.logger.warning("No available cameras detected")

    def stop(self):
        """Stop the camera sensor."""
        self.running = False

        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            try:
                self.capture_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Error stopping capture thread: {e}")

        # Release camera
        if self.cap and not self.simulate:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")

        self.logger.info(f"Stopped camera: {self.name}")

    def get_data(self):
        """
        Get the latest frame from the camera.

        Returns:
            Dictionary with frame data
        """
        current_time = time.time()

        # Check if enough time has passed since the last capture
        if current_time - self.last_capture_time < self.frame_interval:
            return None

        self.last_capture_time = current_time

        if self.simulate:
            # Return simulated data
            description = random.choice(self.simulate_data)

            return {
                'type': self.type,
                'name': self.name,
                'timestamp': current_time,
                'simulated': True,
                'description': description,
                'frame': None
            }
        else:
            try:
                # Get the latest frame from the queue
                try:
                    frame = self.frame_queue.get_nowait()
                    self.frame_queue.task_done()
                    self.current_frame = frame
                except queue.Empty:
                    # If queue is empty, use the current frame
                    frame = self.current_frame

                if frame is None:
                    # No frame available yet
                    return None

                # Resize for vision model and convert to base64
                # The vision model typically expects a smaller image
                vision_frame = cv2.resize(frame, (512, 384))

                # Convert to base64 for transmission to vision model
                _, buffer = cv2.imencode('.jpg', vision_frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')

                return {
                    'type': self.type,
                    'name': self.name,
                    'timestamp': current_time,
                    'simulated': False,
                    'frame': frame,  # Original frame for display
                    'frame_width': frame.shape[1],
                    'frame_height': frame.shape[0],
                    'base64_image': base64_image,  # Base64 encoded image for vision model
                    'description': None  # Will be filled by vision model
                }

            except Exception as e:
                self.logger.error(f"Error getting camera data: {e}")
                return None

    def _capture_loop(self):
        """Background thread for capturing frames."""
        self.logger.debug(f"Starting camera capture loop")

        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    self.logger.error("Camera is not open in capture loop")
                    time.sleep(1.0)
                    continue

                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to capture frame in capture loop")
                    time.sleep(0.1)
                    continue

                # Store frame
                try:
                    # Update current frame
                    self.current_frame = frame

                    # Try to add to queue, but don't block if full
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Queue is full, skip this frame
                    pass

                # Sleep a short time to avoid maxing out CPU
                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in camera capture loop: {e}")
                time.sleep(0.1)

    def test_camera_connection(self):
        """Test if the camera can be accessed and return diagnostic information."""
        test_results = {
            'success': False,
            'camera_device': self.camera_device,
            'errors': [],
            'messages': []
        }

        try:
            # Try to open the camera
            cap = cv2.VideoCapture(self.camera_device)
            test_results['opened'] = cap.isOpened()

            if not cap.isOpened():
                test_results['errors'].append(f"Failed to open camera device {self.camera_device}")
            else:
                test_results['messages'].append(f"Successfully opened camera device {self.camera_device}")

                # Try to read a frame
                ret, frame = cap.read()
                test_results['frame_read'] = ret

                if not ret:
                    test_results['errors'].append("Failed to read frame")
                else:
                    test_results['messages'].append(f"Successfully read frame with shape {frame.shape}")
                    test_results['frame_width'] = frame.shape[1]
                    test_results['frame_height'] = frame.shape[0]
                    test_results['success'] = True

                # Release the camera
                cap.release()

        except Exception as e:
            test_results['errors'].append(f"Exception: {str(e)}")
            test_results['success'] = False

        return test_results

class MicrophoneSensor(SensorInterface):
    """Interface for microphone sensors."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize microphone sensor."""
        super().__init__(config)
        self.type = 'microphone'
        self.data_queue = queue.Queue(maxsize=10)  # Buffer for latest audio
        self.thread = None
        self.capture_interval = config.get('capture_interval', 2.0)  # seconds between audio captures

        # Simulated microphone settings for testing
        self.simulate = config.get('simulate', True)
        self.simulate_data = config.get('simulate_data', [
            "Hello, can you hear me?",
            "What's the weather like today?",
            "I need some help with a problem.",
            "This is a test of the audio system.",
            ""  # Empty for silence
        ])

        self.logger.info(f"Initialized microphone sensor: {self.name}")

    def start(self):
        """Start the microphone sensor."""
        if self.running:
            return

        self.running = True

        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_microphone,
                name=f"microphone_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated microphone: {self.name}")
        else:
            # Implementation for real microphone would go here
            self.logger.warning(f"Real microphone not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation

    def stop(self):
        """Stop the microphone sensor."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.logger.info(f"Stopped microphone: {self.name}")

    def get_data(self) -> Dict[str, Any]:
        """Get the latest audio data."""
        try:
            # Non-blocking get
            data = self.data_queue.get_nowait()
            self.data_queue.task_done()
            return data
        except queue.Empty:
            return {}

    def _simulate_microphone(self):
        """Simulate microphone data for testing."""
        self.logger.debug(f"Starting microphone simulation for {self.name}")

        # Get simulated data if configured
        simulated_data = self.config.get('simulate_data', [
            "Hello, can you hear me?",
            "What's the current time?",
            "Tell me about your capabilities.",
            "What are you thinking about?",
            "What do you see right now?"
        ])

        audio_count = 0

        try:
            while self.running:
                # Only generate audio occasionally to avoid flooding
                if random.random() < 0.3:  # 30% chance each cycle
                    # Generate simulated audio
                    if simulated_data:
                        # Rotate through simulated data
                        audio_text = simulated_data[audio_count % len(simulated_data)]
                        audio_count += 1
                    else:
                        # Default text
                        audio_text = "Simulated audio input."

                    # Create simulated audio data
                    audio_data = {
                        'type': 'microphone',
                        'timestamp': time.time(),
                        'audio_id': audio_count,
                        'text': audio_text,
                        'simulated': True
                    }

                    # Add to data queue
                    self.data_queue.put(audio_data)
                    self.logger.debug(f"Generated simulated audio: {audio_text[:30]}...")

                # Sleep for capture interval
                time.sleep(self.capture_interval)

        except Exception as e:
            self.logger.error(f"Error in microphone simulation: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            self.logger.debug(f"Microphone simulation stopping for {self.name}")

class GenericSensor(SensorInterface):
    """Interface for generic sensors (temperature, motion, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize generic sensor."""
        super().__init__(config)
        self.type = config.get('type', 'generic')
        self.data_queue = queue.Queue(maxsize=5)  # Buffer for latest readings
        self.thread = None
        self.read_interval = config.get('read_interval', 5.0)  # seconds between readings

        # Simulated sensor settings
        self.simulate = config.get('simulate', True)
        self.simulate_min = config.get('simulate_min', 0)
        self.simulate_max = config.get('simulate_max', 100)
        self.simulate_unit = config.get('simulate_unit', '')

        self.logger.info(f"Initialized {self.type} sensor: {self.name}")

    def start(self):
        """Start the sensor."""
        if self.running:
            return

        self.running = True

        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_sensor,
                name=f"{self.type}_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated {self.type} sensor: {self.name}")
        else:
            # Implementation for real sensor would go here
            self.logger.warning(f"Real {self.type} sensor not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation

    def stop(self):
        """Stop the sensor."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.logger.info(f"Stopped {self.type} sensor: {self.name}")

    def get_data(self) -> Dict[str, Any]:
        """Get the latest sensor reading."""
        try:
            # Non-blocking get
            data = self.data_queue.get_nowait()
            self.data_queue.task_done()
            return data
        except queue.Empty:
            return {}

    def _simulate_sensor(self):
        """Simulate sensor data for testing."""
        import random

        self.logger.debug(f"Starting {self.type} sensor simulation for {self.name}")

        while self.running:
            try:
                # Generate a simulated sensor reading
                timestamp = time.time()

                # Generate a random value
                value = random.uniform(self.simulate_min, self.simulate_max)

                # Create sensor data
                sensor_data = {
                    'type': self.type,
                    'timestamp': timestamp,
                    'value': value,
                    'unit': self.simulate_unit,
                    'source': self.name
                }

                # Put in queue, replacing old reading if full
                try:
                    self.data_queue.put_nowait(sensor_data)
                except queue.Full:
                    # Remove old reading
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.task_done()
                    except queue.Empty:
                        pass
                    # Try again
                    self.data_queue.put_nowait(sensor_data)

                # Sleep until next reading
                time.sleep(self.read_interval)

            except Exception as e:
                self.logger.error(f"Error in {self.type} sensor simulation: {e}")
                time.sleep(1.0)  # Sleep longer on error

class SensorManager:
    """
    Manages multiple sensors and provides a unified interface for accessing data.
    """

    def __init__(self, config: Dict[str, Any], platform: str):
        """
        Initialize the sensor manager.

        Args:
            config: Configuration dictionary
            platform: Hardware platform ('mac', 'robot', etc.)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.platform = platform
        self.sensors = {}

        # Initialize sensors based on configuration
        self._initialize_sensors()

        self.logger.info(f"Sensor manager initialized for platform {platform} with {len(self.sensors)} sensors")

    def _initialize_sensors(self):
        """Initialize sensors based on configuration."""
        # Get platform-specific sensors
        platform_sensors = self.config.get(self.platform, self.config.get('default', []))

        for sensor_config in platform_sensors:
            sensor_type = sensor_config.get('type', 'generic')
            sensor_name = sensor_config.get('name', f"{sensor_type}_{len(self.sensors)}")

            # Skip disabled sensors
            if not sensor_config.get('enabled', True):
                self.logger.info(f"Skipping disabled sensor: {sensor_name}")
                continue

            # Create sensor based on type
            try:
                if sensor_type == 'camera':
                    sensor = CameraSensor(sensor_config)
                elif sensor_type == 'microphone':
                    sensor = MicrophoneSensor(sensor_config)
                else:
                    # Generic sensor for everything else
                    sensor = GenericSensor(sensor_config)

                # Add to sensors dictionary
                self.sensors[sensor_name] = sensor
                self.logger.info(f"Added sensor: {sensor_name} ({sensor_type})")

            except Exception as e:
                self.logger.error(f"Error initializing sensor {sensor_name}: {e}")

    def start(self):
        """Start all sensors."""
        for name, sensor in self.sensors.items():
            try:
                sensor.start()
            except Exception as e:
                self.logger.error(f"Error starting sensor {name}: {e}")

    def stop(self):
        """Stop all sensors."""
        for name, sensor in self.sensors.items():
            try:
                sensor.stop()
            except Exception as e:
                self.logger.error(f"Error stopping sensor {name}: {e}")

    def get_data(self, sensor_name: str) -> Dict[str, Any]:
        """
        Get data from a specific sensor.

        Args:
            sensor_name: Name of the sensor

        Returns:
            Sensor data dictionary
        """
        sensor = self.sensors.get(sensor_name)
        if not sensor:
            self.logger.warning(f"Sensor not found: {sensor_name}")
            return {}

        return sensor.get_data()

    def get_all_data(self) -> Dict[str, Any]:
        """
        Get data from all sensors.

        Returns:
            Dictionary of sensor data by sensor type
        """
        result = {}

        for name, sensor in self.sensors.items():
            data = sensor.get_data()
            if data:  # Only include if there's data
                sensor_type = data.get('type', 'unknown')

                # Group by sensor type
                if sensor_type not in result:
                    result[sensor_type] = []

                result[sensor_type].append(data)

        return result

    def get_sensors_by_type(self, sensor_type: str) -> List[str]:
        """
        Get a list of sensor names of a specific type.

        Args:
            sensor_type: Type of sensors to find

        Returns:
            List of sensor names
        """
        return [name for name, sensor in self.sensors.items()
                if sensor.type == sensor_type]
