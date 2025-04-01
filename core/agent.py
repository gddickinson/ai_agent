"""
Agent Core Module
Handles the orchestration of all agent components.
"""

import logging
import traceback
import threading
import time
from typing import Dict, Any

from core.consciousness import ConsciousnessModule
from core.perception import PerceptionModule
from core.cognition import CognitionModule
from core.memory import MemoryManager
from core.autonomy import AutonomyModule
from hardware.sensors import SensorManager
from hardware.actuators import ActuatorManager
from llm.manager import LLMManager

class EmbodiedAgent:
    """
    Main agent class that coordinates perception, cognition, consciousness,
    and interaction with hardware.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent with configuration.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.is_running = False
        self._threads = []

        # Initialize LLM manager
        self.llm_manager = LLMManager(config['llm'])

        # Initialize memory systems
        self.memory = MemoryManager(config['memory'])

        # Initialize hardware interfaces based on platform
        platform = config['hardware']['platform']
        self.logger.info(f"Initializing hardware for platform: {platform}")

        # Set up sensor and actuator managers
        self.sensors = SensorManager(config['hardware']['sensors'], platform)
        self.actuators = ActuatorManager(config['hardware']['actuators'], platform)

        # Initialize cognitive modules
        self.perception = PerceptionModule(
            config['perception'],
            self.llm_manager,
            self.memory
        )

        self.cognition = CognitionModule(
            config['cognition'],
            self.llm_manager,
            self.memory
        )

        self.consciousness = ConsciousnessModule(
            config.get('consciousness', {}),
            self.llm_manager,
            self.memory,
            self.actuators
        )

        # Initialize autonomy module if configured
        if 'autonomy' in config:
            self.autonomy = AutonomyModule(
                config['autonomy'],
                self.llm_manager,
                self.memory,
                self.consciousness
            )
            # Connect autonomy module to consciousness
            self.consciousness.set_autonomy_module(self.autonomy)
        else:
            self.autonomy = None
            self.logger.info("Autonomy module not configured, running without autonomous thinking")

        # Register callbacks for message handling
        self.consciousness.register_message_callback(
            'sent',
            lambda message: self.logger.info(f"Message sent: {message.get('content', '')[:100]}...")
        )

        self.logger.info("Agent initialized successfully")


    def start(self):
        """Start all agent modules and processing threads."""
        if self.is_running:
            self.logger.warning("Agent is already running")
            return

        self.logger.info("Starting agent")

        # Apply memory decay (clean up old, low-importance memories)
        try:
            decayed_count = self.memory.decay_memories(threshold_days=30)
            if decayed_count > 0:
                self.logger.info(f"Decayed {decayed_count} old memories")
        except Exception as e:
            self.logger.error(f"Error decaying memories: {e}")

        # Start hardware systems
        self.sensors.start()
        self.actuators.start()

        # Start memory system
        self.memory.start()

        # Start the cognitive processing pipeline
        self._start_perception_processing()
        self._start_cognition_processing()
        self._start_consciousness_processing()

        # Start autonomy module if available
        if self.autonomy:
            self.autonomy.start()
            self.logger.info("Autonomy module started")

        self.is_running = True
        self.logger.info("Agent started successfully")

    def stop(self):
        """Stop all agent modules and processing threads."""
        if not self.is_running:
            return

        self.logger.info("Stopping agent")
        self.is_running = False

        # Stop all threads
        for thread in self._threads:
            if thread.is_alive():
                # Give threads a chance to stop gracefully
                thread.join(timeout=2.0)

        # Stop components
        if self.autonomy:
            self.autonomy.stop()
        self.consciousness.stop()
        self.cognition.stop()
        self.perception.stop()
        self.memory.stop()
        self.actuators.stop()
        self.sensors.stop()

        self.logger.info("Agent stopped")



    def _start_perception_processing(self):
        """Start the perception processing thread."""
        def perception_loop():
            self.logger.debug("Starting perception loop")
            perception_count = 0
            while self.is_running:
                try:
                    # Get sensor data
                    sensor_data = self.sensors.get_all_data()

                    # Log the sensor data for debugging
                    if sensor_data:
                        perception_count += 1
                        self.logger.debug(f"Processing sensor data from {len(sensor_data)} sensors (#{perception_count})")
                        for sensor_name, data in sensor_data.items():
                            sensor_type = data.get('type', 'unknown')
                            self.logger.debug(f"Sensor {sensor_name} ({sensor_type}): {str(data)[:100]}...")

                            # Store basic information in working memory for immediate access
                            if sensor_type == 'camera' and 'description' in data:
                                self.memory.add_to_working_memory(
                                    item={
                                        'type': 'perception',
                                        'sensor': sensor_name,
                                        'content': f"Visual: {data['description']}",
                                        'timestamp': data.get('timestamp', time.time())
                                    },
                                    importance=0.6
                                )
                            elif sensor_type == 'microphone' and 'text' in data:
                                self.memory.add_to_working_memory(
                                    item={
                                        'type': 'perception',
                                        'sensor': sensor_name,
                                        'content': f"Audio: {data['text']}",
                                        'timestamp': data.get('timestamp', time.time())
                                    },
                                    importance=0.7
                                )

                    # Process perception
                    if sensor_data:
                        self.perception.process(sensor_data)

                    # Sleep to avoid excessive CPU usage
                    time.sleep(self.config['processing']['perception_interval'])

                except Exception as e:
                    self.logger.error(f"Error in perception loop: {e}")
                    self.logger.error(traceback.format_exc())
                    time.sleep(1.0)  # Sleep a bit longer on error

            self.logger.debug("Perception loop stopped")

        perception_thread = threading.Thread(
            target=perception_loop,
            name="perception_thread",
            daemon=True
        )
        perception_thread.start()
        self._threads.append(perception_thread)

    def _start_cognition_processing(self):
        """Start the cognition processing thread."""
        def cognition_loop():
            self.logger.debug("Starting cognition loop")
            while self.is_running:
                try:
                    # Process cognition based on latest perceptions
                    self.cognition.process()

                    # Sleep to avoid excessive CPU usage
                    time.sleep(self.config['processing']['cognition_interval'])

                except Exception as e:
                    self.logger.error(f"Error in cognition loop: {e}")
                    time.sleep(1.0)  # Sleep a bit longer on error

            self.logger.debug("Cognition loop stopped")

        cognition_thread = threading.Thread(
            target=cognition_loop,
            name="cognition_thread",
            daemon=True
        )
        cognition_thread.start()
        self._threads.append(cognition_thread)

    def _start_consciousness_processing(self):
        """Start the consciousness processing thread."""
        def consciousness_loop():
            self.logger.debug("Starting consciousness loop")
            while self.is_running:
                try:
                    # Process consciousness layer
                    self.consciousness.process()

                    # Sleep to avoid excessive CPU usage
                    time.sleep(self.config['processing']['consciousness_interval'])

                except Exception as e:
                    self.logger.error(f"Error in consciousness loop: {e}")
                    time.sleep(1.0)  # Sleep a bit longer on error

            self.logger.debug("Consciousness loop stopped")

        consciousness_thread = threading.Thread(
            target=consciousness_loop,
            name="consciousness_thread",
            daemon=True
        )
        consciousness_thread.start()
        self._threads.append(consciousness_thread)
