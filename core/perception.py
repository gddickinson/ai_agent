"""
Perception Module
Handles processing of sensory inputs and their interpretation.
"""

import logging
import time
import json
from typing import Dict, Any, List

from llm.manager import LLMManager
from core.memory import MemoryManager

class PerceptionModule:
    """
    Processes and interprets raw sensory data into meaningful perceptions.
    Uses specialized LLMs for different sensory modalities.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        llm_manager: LLMManager,
        memory_manager: MemoryManager
    ):
        """
        Initialize the perception module.
        
        Args:
            config: Configuration dictionary
            llm_manager: LLM manager instance
            memory_manager: Memory manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm_manager = llm_manager
        self.memory = memory_manager
        
        # Configure perception LLMs
        self.vision_model = config.get('vision_model', 'vision_interpreter')
        self.audio_model = config.get('audio_model', 'audio_interpreter')
        self.general_model = config.get('general_model', 'general_interpreter')
        
        # Last processed timestamp for each sensor type
        self.last_processed = {}
        
        # Processing intervals
        self.process_intervals = config.get('process_intervals', {
            'camera': 1.0,  # Process camera data every 1 second
            'microphone': 0.5,  # Process audio data every 0.5 seconds
            'lidar': 2.0,  # Process lidar data every 2 seconds
            'general': 0.2,  # General sensor data every 0.2 seconds
        })
        
        self.logger.info("Perception module initialized")
    
    def process(self, sensor_data: Dict[str, Any]):
        """
        Process incoming sensor data.
        
        Args:
            sensor_data: Dictionary of sensor data by sensor type
        """
        current_time = time.time()
        
        # Process different sensor types with appropriate models
        for sensor_type, data in sensor_data.items():
            # Skip if we processed this sensor type recently
            last_time = self.last_processed.get(sensor_type, 0)
            interval = self.process_intervals.get(sensor_type, 
                                                 self.process_intervals.get('general', 1.0))
            
            if current_time - last_time < interval:
                continue
                
            self.last_processed[sensor_type] = current_time
            
            if sensor_type == 'camera':
                self._process_vision(data)
            elif sensor_type == 'microphone':
                self._process_audio(data)
            else:
                self._process_general_sensor(sensor_type, data)
    
    def _process_vision(self, image_data: Dict[str, Any]):
        """
        Process vision data using vision LLM.
        
        Args:
            image_data: Image data dictionary
        """
        self.logger.debug("Processing vision data")
        
        # Extract metadata
        timestamp = image_data.get('timestamp', time.time())
        image_format = image_data.get('format', 'base64')
        image_content = image_data.get('content', '')
        
        # Skip if no content
        if not image_content:
            self.logger.warning("Empty image content received")
            return
        
        # Create prompt for vision interpretation
        prompt = self._create_vision_prompt(image_content, image_format, timestamp)
        
        # Send to LLM for processing
        def vision_callback(result, task):
            if result:
                try:
                    # Extract interpretation
                    interpretation = result.get('text', '')
                    
                    # Create structured perception
                    perception = {
                        'type': 'vision',
                        'timestamp': timestamp,
                        'raw_data': {
                            'format': image_format,
                            # Don't store full image in perception, too large
                            'summary': f"Image captured at {timestamp}"
                        },
                        'interpretation': interpretation,
                        'processed_at': time.time()
                    }
                    
                    # Store in perception memory
                    self.memory.add_perception(perception)
                    
                    self.logger.debug(f"Processed vision perception: {len(interpretation)} chars")
                except Exception as e:
                    self.logger.error(f"Error processing vision result: {e}")
            else:
                self.logger.error(f"Vision processing failed: {task.error}")
        
        # Submit task
        self.llm_manager.submit_task(
            model_name=self.vision_model,
            prompt=prompt,
            callback=vision_callback,
            max_tokens=1024,
            temperature=0.2
        )
    
    def _process_audio(self, audio_data: Dict[str, Any]):
        """
        Process audio data using audio LLM.
        
        Args:
            audio_data: Audio data dictionary
        """
        self.logger.debug("Processing audio data")
        
        # Extract metadata
        timestamp = audio_data.get('timestamp', time.time())
        audio_format = audio_data.get('format', 'text')  # Assume pre-transcribed
        audio_content = audio_data.get('content', '')
        
        # Skip if no content
        if not audio_content:
            self.logger.warning("Empty audio content received")
            return
        
        # Create prompt for audio interpretation
        prompt = self._create_audio_prompt(audio_content, audio_format, timestamp)
        
        # Send to LLM for processing
        def audio_callback(result, task):
            if result:
                try:
                    # Extract interpretation
                    interpretation = result.get('text', '')
                    
                    # Create structured perception
                    perception = {
                        'type': 'audio',
                        'timestamp': timestamp,
                        'raw_data': {
                            'format': audio_format,
                            'content': audio_content if audio_format == 'text' else 
                                      f"Audio captured at {timestamp}"
                        },
                        'interpretation': interpretation,
                        'processed_at': time.time()
                    }
                    
                    # Store in perception memory
                    self.memory.add_perception(perception)
                    
                    self.logger.debug(f"Processed audio perception: {len(interpretation)} chars")
                except Exception as e:
                    self.logger.error(f"Error processing audio result: {e}")
            else:
                self.logger.error(f"Audio processing failed: {task.error}")
        
        # Submit task
        self.llm_manager.submit_task(
            model_name=self.audio_model,
            prompt=prompt,
            callback=audio_callback,
            max_tokens=1024,
            temperature=0.3
        )
    
    def _process_general_sensor(self, sensor_type: str, sensor_data: Dict[str, Any]):
        """
        Process general sensor data.
        
        Args:
            sensor_type: Type of sensor
            sensor_data: Sensor data dictionary
        """
        self.logger.debug(f"Processing {sensor_type} data")
        
        # Extract metadata
        timestamp = sensor_data.get('timestamp', time.time())
        
        # Create a summarized version for the prompt
        data_summary = json.dumps(sensor_data, indent=2)
        
        # Create prompt for general sensor interpretation
        prompt = self._create_general_sensor_prompt(sensor_type, data_summary, timestamp)
        
        # Send to LLM for processing
        def sensor_callback(result, task):
            if result:
                try:
                    # Extract interpretation
                    interpretation = result.get('text', '')
                    
                    # Create structured perception
                    perception = {
                        'type': sensor_type,
                        'timestamp': timestamp,
                        'raw_data': {
                            'summary': f"{sensor_type} data captured at {timestamp}"
                        },
                        'interpretation': interpretation,
                        'processed_at': time.time()
                    }
                    
                    # Store in perception memory
                    self.memory.add_perception(perception)
                    
                    self.logger.debug(f"Processed {sensor_type} perception")
                except Exception as e:
                    self.logger.error(f"Error processing {sensor_type} result: {e}")
            else:
                self.logger.error(f"{sensor_type} processing failed: {task.error}")
        
        # Submit task
        self.llm_manager.submit_task(
            model_name=self.general_model,
            prompt=prompt,
            callback=sensor_callback,
            max_tokens=512,
            temperature=0.3
        )
    
    def _create_vision_prompt(self, image_content: str, image_format: str, timestamp: float) -> str:
        """Create a prompt for vision interpretation."""
        # In a real implementation, we would embed the image or provide a URL
        # For now, we'll assume image_content is a descriptive text (stand-in for actual image data)
        
        prompt = f"""
        # Vision Interpretation Task
        
        You are the visual perception system of an embodied AI agent. Your task is to interpret 
        the visual input and describe what you see in detail. Focus on:
        
        - Objects, people, and environments
        - Spatial relationships
        - Activities and movements
        - Text content if visible
        - Emotional expressions if people are present
        
        Please provide a clear, factual description of what is visible in the scene.
        
        ## Image Information
        - Timestamp: {timestamp}
        - Format: {image_format}
        
        ## Image Content
        {image_content[:500]}... (content truncated for prompt)
        
        ## Response Format
        Provide a detailed description in JSON format with these keys:
        - 'scene_overview': Brief overall description
        - 'key_elements': List of main items/people
        - 'spatial_layout': Description of layout
        - 'actions': Any activities detected
        - 'text_content': Any visible text
        - 'emotional_tone': Emotional assessment
        """
        
        return prompt
    
    def _create_audio_prompt(self, audio_content: str, audio_format: str, timestamp: float) -> str:
        """Create a prompt for audio interpretation."""
        prompt = f"""
        # Audio Interpretation Task
        
        You are the auditory perception system of an embodied AI agent. Your task is to interpret 
        the audio input and describe what you hear in detail. Focus on:
        
        - Speech content and speakers
        - Environmental sounds
        - Emotional tone
        - Background noise
        - Music if present
        
        ## Audio Information
        - Timestamp: {timestamp}
        - Format: {audio_format}
        
        ## Audio Content
        {audio_content}
        
        ## Response Format
        Provide a detailed description in JSON format with these keys:
        - 'audio_overview': Brief overall description
        - 'speech_content': Transcription or summary of speech
        - 'speakers': Number and characteristics of speakers
        - 'background_sounds': Description of background noises
        - 'emotional_tone': Emotional assessment
        - 'confidence': Your confidence in this interpretation (low/medium/high)
        """
        
        return prompt
    
    def _create_general_sensor_prompt(self, sensor_type: str, data_summary: str, timestamp: float) -> str:
        """Create a prompt for general sensor interpretation."""
        prompt = f"""
        # Sensor Data Interpretation Task
        
        You are the sensory perception system of an embodied AI agent. Your task is to interpret 
        data from a {sensor_type} sensor and extract meaningful information.
        
        ## Sensor Information
        - Sensor Type: {sensor_type}
        - Timestamp: {timestamp}
        
        ## Sensor Data
        ```
        {data_summary}
        ```
        
        ## Response Format
        Provide a detailed interpretation in JSON format with these keys:
        - 'data_overview': Brief summary of what this data represents
        - 'key_readings': Most important values or patterns
        - 'anomalies': Any unusual readings or patterns
        - 'implications': What this data suggests about the environment
        - 'confidence': Your confidence in this interpretation (low/medium/high)
        """
        
        return prompt
    
    def stop(self):
        """Stop the perception module."""
        self.logger.info("Stopping perception module")
        # Nothing to stop in this implementation
