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

class CameraSensor(SensorInterface):
    """Interface for camera input."""

    def __init__(self, config):
        """Initialize camera sensor."""
        super().__init__(config)
        self.type = 'camera'
        self.data_queue = queue.Queue()
        self.thread = None

        # Camera settings
        self.frame_interval = config.get('frame_interval', 1.0)  # seconds between frames

        # Simulation settings
        self.simulate = config.get('simulate', False)
        self.simulate_data = config.get('simulate_data', [])

        # Initialize frame counter
        self.frame_count = 0

        self.logger.info(f"Initialized camera sensor: {self.name}")

    def start(self):
        """Start the camera sensor."""
        if self.running:
            return

        self.running = True

        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_camera,
                name=f"camera_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated camera: {self.name}")
        else:
            # Implementation for real camera would go here
            # e.g., initialize webcam, start capture thread, etc.
            self.logger.warning(f"Real camera not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation

    def stop(self):
        """Stop the camera sensor."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.logger.info(f"Stopped camera: {self.name}")

    def get_data(self) -> Dict[str, Any]:
        """Get the latest camera frame."""
        try:
            # Non-blocking get
            data = self.data_queue.get_nowait()
            self.data_queue.task_done()
            return data
        except queue.Empty:
            return {}

    # Update in sensors.py for the CameraSensor class
    def _simulate_camera(self):
        """Simulate camera data for testing."""
        self.logger.debug(f"Starting camera simulation for {self.name}")

        # Get simulated data if configured
        simulated_data = self.config.get('simulate_data', [
            "A desk with a computer monitor, keyboard, and mouse.",
            "A room with various furniture and equipment.",
            "A person sitting at a desk working on a computer.",
            "An office space with natural light coming through windows."
        ])

        try:
            while self.running:
                # Generate simulated frame
                if simulated_data:
                    # Rotate through simulated data
                    frame_description = simulated_data[self.frame_count % len(simulated_data)]
                    self.frame_count += 1
                else:
                    # Default description
                    frame_description = "A room with various objects."

                # Create simulated frame data
                frame_data = {
                    'type': 'camera',
                    'timestamp': time.time(),
                    'frame_id': self.frame_count,
                    'description': frame_description,
                    'simulated': True
                }

                # Add to data queue
                self.data_queue.put(frame_data)
                self.logger.debug(f"Generated simulated camera frame: {frame_description[:30]}...")

                # Sleep for frame interval
                time.sleep(self.frame_interval)

        except Exception as e:
            self.logger.error(f"Error in camera simulation: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            self.logger.debug(f"Camera simulation stopping for {self.name}")

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
