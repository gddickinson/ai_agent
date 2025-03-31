"""
Actuators Module
Handles interfacing with various output hardware on different platforms.
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import json
import os

class ActuatorInterface(ABC):
    """Base class for actuator interfaces."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the actuator interface.
        
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
        """Start the actuator."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the actuator."""
        pass
    
    @abstractmethod
    def execute(self, command: Dict[str, Any]) -> bool:
        """
        Execute a command on this actuator.
        
        Args:
            command: Command dictionary
            
        Returns:
            Success flag
        """
        pass

class DisplayActuator(ActuatorInterface):
    """Interface for display output."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize display actuator."""
        super().__init__(config)
        self.type = 'display'
        self.output_queue = queue.Queue()
        self.thread = None
        
        # Simulated display settings
        self.simulate = config.get('simulate', True)
        self.output_file = config.get('output_file', None)
        
        self.logger.info(f"Initialized display actuator: {self.name}")
    
    def start(self):
        """Start the display actuator."""
        if self.running:
            return
        
        self.running = True
        
        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_display,
                name=f"display_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated display: {self.name}")
        else:
            # Implementation for real display would go here
            self.logger.warning(f"Real display not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation
    
    def stop(self):
        """Stop the display actuator."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.logger.info(f"Stopped display: {self.name}")
    
    def execute(self, command: Dict[str, Any]) -> bool:
        """
        Display content on the output device.
        
        Args:
            command: Command with content to display
            
        Returns:
            Success flag
        """
        # Add to output queue
        try:
            self.output_queue.put(command, timeout=1.0)
            return True
        except queue.Full:
            self.logger.error(f"Display output queue full, dropping command")
            return False
    
    def _simulate_display(self):
        """Simulate display output for testing."""
        self.logger.debug(f"Starting display simulation for {self.name}")
        
        # Open output file if specified
        output_file = None
        if self.output_file:
            try:
                # Make sure directory exists
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                
                # Open for append
                output_file = open(self.output_file, 'a')
                self.logger.info(f"Opened display output file: {self.output_file}")
            except Exception as e:
                self.logger.error(f"Error opening display output file: {e}")
        
        try:
            while self.running:
                try:
                    # Get next output
                    command = self.output_queue.get(timeout=0.5)
                    
                    # Process output
                    content = command.get('content', '')
                    content_type = command.get('content_type', 'text')
                    
                    # Log the output
                    self.logger.info(f"DISPLAY OUTPUT ({content_type}): {content[:100]}...")
                    
                    # Write to file if open
                    if output_file:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        output_file.write(f"\n[{timestamp}] {content}\n")
                        output_file.flush()
                    
                    # Mark as done
                    self.output_queue.task_done()
                    
                except queue.Empty:
                    # No output to process
                    pass
                except Exception as e:
                    self.logger.error(f"Error in display simulation: {e}")
                    time.sleep(0.1)  # Prevent thrashing on error
        
        finally:
            # Close output file if open
            if output_file:
                output_file.close()
                self.logger.info(f"Closed display output file")

class SpeakerActuator(ActuatorInterface):
    """Interface for audio output."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize speaker actuator."""
        super().__init__(config)
        self.type = 'speaker'
        self.output_queue = queue.Queue()
        self.thread = None
        
        # Simulated speaker settings
        self.simulate = config.get('simulate', True)
        self.output_file = config.get('output_file', None)
        
        self.logger.info(f"Initialized speaker actuator: {self.name}")
    
    def start(self):
        """Start the speaker actuator."""
        if self.running:
            return
        
        self.running = True
        
        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_speaker,
                name=f"speaker_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated speaker: {self.name}")
        else:
            # Implementation for real speaker would go here
            self.logger.warning(f"Real speaker not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation
    
    def stop(self):
        """Stop the speaker actuator."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.logger.info(f"Stopped speaker: {self.name}")
    
    def execute(self, command: Dict[str, Any]) -> bool:
        """
        Play audio through the speaker.
        
        Args:
            command: Command with audio content
            
        Returns:
            Success flag
        """
        # Add to output queue
        try:
            self.output_queue.put(command, timeout=1.0)
            return True
        except queue.Full:
            self.logger.error(f"Speaker output queue full, dropping command")
            return False
    
    def _simulate_speaker(self):
        """Simulate speaker output for testing."""
        self.logger.debug(f"Starting speaker simulation for {self.name}")
        
        # Open output file if specified
        output_file = None
        if self.output_file:
            try:
                # Make sure directory exists
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                
                # Open for append
                output_file = open(self.output_file, 'a')
                self.logger.info(f"Opened speaker output file: {self.output_file}")
            except Exception as e:
                self.logger.error(f"Error opening speaker output file: {e}")
        
        try:
            while self.running:
                try:
                    # Get next output
                    command = self.output_queue.get(timeout=0.5)
                    
                    # Process output
                    content = command.get('content', '')
                    
                    # Log the output
                    self.logger.info(f"SPEAKER OUTPUT: {content[:100]}...")
                    
                    # Write to file if open
                    if output_file:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        output_file.write(f"\n[{timestamp}] SPEECH: {content}\n")
                        output_file.flush()
                    
                    # Mark as done
                    self.output_queue.task_done()
                    
                except queue.Empty:
                    # No output to process
                    pass
                except Exception as e:
                    self.logger.error(f"Error in speaker simulation: {e}")
                    time.sleep(0.1)  # Prevent thrashing on error
        
        finally:
            # Close output file if open
            if output_file:
                output_file.close()
                self.logger.info(f"Closed speaker output file")

class MotorActuator(ActuatorInterface):
    """Interface for motor/movement control."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize motor actuator."""
        super().__init__(config)
        self.type = 'motor'
        self.command_queue = queue.Queue()
        self.thread = None
        
        # Simulated motor settings
        self.simulate = config.get('simulate', True)
        self.motor_type = config.get('motor_type', 'generic')  # 'wheel', 'arm', etc.
        
        self.logger.info(f"Initialized {self.motor_type} motor actuator: {self.name}")
    
    def start(self):
        """Start the motor actuator."""
        if self.running:
            return
        
        self.running = True
        
        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_motor,
                name=f"motor_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated {self.motor_type} motor: {self.name}")
        else:
            # Implementation for real motor would go here
            self.logger.warning(f"Real motor not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation
    
    def stop(self):
        """Stop the motor actuator."""
        self.running = False
        
        # Send stop command to ensure motor stops
        try:
            self.command_queue.put({
                'command': 'stop',
                'priority': 100  # High priority
            }, timeout=1.0)
        except queue.Full:
            self.logger.error(f"Motor command queue full, couldn't send stop command")
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        self.logger.info(f"Stopped {self.motor_type} motor: {self.name}")
    
    def execute(self, command: Dict[str, Any]) -> bool:
        """
        Execute a motor command.
        
        Args:
            command: Motor command dictionary
            
        Returns:
            Success flag
        """
        # Add to command queue
        try:
            # Default priority is 50 (mid-range)
            command['priority'] = command.get('priority', 50)
            self.command_queue.put(command, timeout=1.0)
            return True
        except queue.Full:
            self.logger.error(f"Motor command queue full, dropping command")
            return False
    
    def _simulate_motor(self):
        """Simulate motor actions for testing."""
        self.logger.debug(f"Starting {self.motor_type} motor simulation for {self.name}")
        
        # Current motor state
        current_speed = 0.0
        current_position = 0.0
        target_position = 0.0
        
        try:
            while self.running:
                try:
                    # Get next command (non-blocking)
                    try:
                        command = self.command_queue.get_nowait()
                        
                        # Process command
                        cmd_type = command.get('command', 'stop')
                        
                        if cmd_type == 'stop':
                            current_speed = 0.0
                            self.logger.info(f"Motor {self.name} stopped")
                        
                        elif cmd_type == 'move':
                            speed = command.get('speed', 0.5)  # 0.0-1.0
                            direction = command.get('direction', 'forward')
                            
                            # Set speed and direction
                            if direction == 'forward':
                                current_speed = speed
                            elif direction == 'backward':
                                current_speed = -speed
                            else:
                                current_speed = 0.0
                                
                            self.logger.info(f"Motor {self.name} moving {direction} at speed {speed}")
                        
                        elif cmd_type == 'position':
                            target_position = command.get('position', current_position)
                            speed = command.get('speed', 0.5)  # 0.0-1.0
                            
                            # Set speed based on direction needed
                            if target_position > current_position:
                                current_speed = speed
                            elif target_position < current_position:
                                current_speed = -speed
                            else:
                                current_speed = 0.0
                                
                            self.logger.info(f"Motor {self.name} moving to position {target_position}")
                        
                        # Mark command as done
                        self.command_queue.task_done()
                        
                    except queue.Empty:
                        # No command to process
                        pass
                    
                    # Update position based on speed
                    if current_speed != 0.0:
                        current_position += current_speed * 0.1  # Simple simulation
                        
                        # Check if reached target for position commands
                        if (current_speed > 0 and current_position >= target_position) or \
                           (current_speed < 0 and current_position <= target_position):
                            current_position = target_position
                            current_speed = 0.0
                            self.logger.info(f"Motor {self.name} reached position {current_position}")
                    
                    # Sleep a bit
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error in motor simulation: {e}")
                    time.sleep(0.1)  # Prevent thrashing on error
        
        finally:
            # Ensure motor stops
            self.logger.info(f"Motor {self.name} simulation stopping")

class GenericActuator(ActuatorInterface):
    """Interface for generic actuators."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize generic actuator."""
        super().__init__(config)
        self.command_queue = queue.Queue()
        self.thread = None
        
        # Simulated actuator settings
        self.simulate = config.get('simulate', True)
        
        self.logger.info(f"Initialized {self.type} actuator: {self.name}")
    
    def start(self):
        """Start the actuator."""
        if self.running:
            return
        
        self.running = True
        
        if self.simulate:
            self.thread = threading.Thread(
                target=self._simulate_actuator,
                name=f"{self.type}_{self.name}_thread",
                daemon=True
            )
            self.thread.start()
            self.logger.info(f"Started simulated {self.type} actuator: {self.name}")
        else:
            # Implementation for real actuator would go here
            self.logger.warning(f"Real {self.type} actuator not implemented, falling back to simulation")
            self.simulate = True
            self.start()  # Restart with simulation
    
    def stop(self):
        """Stop the actuator."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.logger.info(f"Stopped {self.type} actuator: {self.name}")
    
    def execute(self, command: Dict[str, Any]) -> bool:
        """
        Execute a command on this actuator.
        
        Args:
            command: Command dictionary
            
        Returns:
            Success flag
        """
        # Add to command queue
        try:
            self.command_queue.put(command, timeout=1.0)
            return True
        except queue.Full:
            self.logger.error(f"Actuator command queue full, dropping command")
            return False
    
    def _simulate_actuator(self):
        """Simulate actuator for testing."""
        self.logger.debug(f"Starting {self.type} actuator simulation for {self.name}")
        
        try:
            while self.running:
                try:
                    # Get next command
                    command = self.command_queue.get(timeout=0.5)
                    
                    # Log the command
                    self.logger.info(f"ACTUATOR {self.name} ({self.type}) COMMAND: {command}")
                    
                    # Simulate processing time
                    time.sleep(0.2)
                    
                    # Mark as done
                    self.command_queue.task_done()
                    
                except queue.Empty:
                    # No command to process
                    pass
                except Exception as e:
                    self.logger.error(f"Error in actuator simulation: {e}")
                    time.sleep(0.1)  # Prevent thrashing on error
        
        finally:
            self.logger.info(f"Actuator {self.name} simulation stopping")

class ActuatorManager:
    """
    Manages multiple actuators and provides a unified interface for controlling them.
    """
    
    def __init__(self, config: Dict[str, Any], platform: str):
        """
        Initialize the actuator manager.
        
        Args:
            config: Configuration dictionary
            platform: Hardware platform ('mac', 'robot', etc.)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.platform = platform
        self.actuators = {}
        
        # Initialize actuators based on configuration
        self._initialize_actuators()
        
        self.logger.info(f"Actuator manager initialized for platform {platform} with {len(self.actuators)} actuators")
    
    def _initialize_actuators(self):
        """Initialize actuators based on configuration."""
        # Get platform-specific actuators
        platform_actuators = self.config.get(self.platform, self.config.get('default', []))
        
        for actuator_config in platform_actuators:
            actuator_type = actuator_config.get('type', 'generic')
            actuator_name = actuator_config.get('name', f"{actuator_type}_{len(self.actuators)}")
            
            # Skip disabled actuators
            if not actuator_config.get('enabled', True):
                self.logger.info(f"Skipping disabled actuator: {actuator_name}")
                continue
            
            # Create actuator based on type
            try:
                if actuator_type == 'display':
                    actuator = DisplayActuator(actuator_config)
                elif actuator_type == 'speaker':
                    actuator = SpeakerActuator(actuator_config)
                elif actuator_type == 'motor':
                    actuator = MotorActuator(actuator_config)
                else:
                    # Generic actuator for everything else
                    actuator = GenericActuator(actuator_config)
                
                # Add to actuators dictionary
                self.actuators[actuator_name] = actuator
                self.logger.info(f"Added actuator: {actuator_name} ({actuator_type})")
                
            except Exception as e:
                self.logger.error(f"Error initializing actuator {actuator_name}: {e}")
    
    def start(self):
        """Start all actuators."""
        for name, actuator in self.actuators.items():
            try:
                actuator.start()
            except Exception as e:
                self.logger.error(f"Error starting actuator {name}: {e}")
    
    def stop(self):
        """Stop all actuators."""
        for name, actuator in self.actuators.items():
            try:
                actuator.stop()
            except Exception as e:
                self.logger.error(f"Error stopping actuator {name}: {e}")
    
    def execute_action(self, action_type: str, target: str, parameters: Dict[str, Any]) -> bool:
        """
        Execute an action using appropriate actuators.
        
        Args:
            action_type: Type of action
            target: Target actuator or system
            parameters: Action parameters
            
        Returns:
            Success flag
        """
        self.logger.debug(f"Executing action: {action_type} on {target}")
        
        success = False
        
        # Handle based on action type
        if action_type == 'move':
            # Movement action
            motors = self.get_actuators_by_type('motor')
            if not motors:
                self.logger.warning(f"No motor actuators available for move action")
                return False
                
            # Execute on all motor actuators or specific ones
            for motor_name in motors:
                if target == 'all' or target == motor_name:
                    self.actuators[motor_name].execute({
                        'command': 'move',
                        'direction': parameters.get('direction', 'forward'),
                        'speed': parameters.get('speed', 0.5)
                    })
                    success = True
        
        elif action_type == 'speak':
            # Voice output
            speakers = self.get_actuators_by_type('speaker')
            if not speakers:
                self.logger.warning(f"No speaker actuators available for speak action")
                return False
                
            # Execute on all speakers or specific ones
            for speaker_name in speakers:
                if target == 'all' or target == speaker_name:
                    self.actuators[speaker_name].execute({
                        'content': parameters.get('text', ''),
                        'volume': parameters.get('volume', 1.0),
                        'rate': parameters.get('rate', 1.0)
                    })
                    success = True
        
        elif action_type == 'display':
            # Visual output
            displays = self.get_actuators_by_type('display')
            if not displays:
                self.logger.warning(f"No display actuators available for display action")
                return False
                
            # Execute on all displays or specific ones
            for display_name in displays:
                if target == 'all' or target == display_name:
                    self.actuators[display_name].execute({
                        'content': parameters.get('text', ''),
                        'content_type': parameters.get('content_type', 'text')
                    })
                    success = True
        
        else:
            # Try to find actuator by name
            actuator = self.actuators.get(target)
            if actuator:
                result = actuator.execute({
                    'command': action_type,
                    **parameters
                })
                success = result
            else:
                self.logger.warning(f"Unknown action type or target: {action_type} / {target}")
        
        return success
    
    def communicate(self, message: Dict[str, Any]) -> bool:
        """
        Send a communication message through appropriate actuators.
        
        Args:
            message: Message dictionary
            
        Returns:
            Success flag
        """
        self.logger.debug(f"Communicating message: {message.get('id', 'unknown')}")
        
        content = message.get('content', '')
        if not content:
            self.logger.warning(f"Empty message content")
            return False
            
        success = False
        
        # Send to displays
        displays = self.get_actuators_by_type('display')
        for display_name in displays:
            result = self.actuators[display_name].execute({
                'content': content,
                'content_type': 'text',
                'message_id': message.get('id')
            })
            success = success or result
        
        # Send to speakers if speech output enabled
        if message.get('speech_output', True):
            speakers = self.get_actuators_by_type('speaker')
            for speaker_name in speakers:
                result = self.actuators[speaker_name].execute({
                    'content': content,
                    'volume': message.get('volume', 1.0),
                    'rate': message.get('rate', 1.0)
                })
                success = success or result
        
        return success
    
    def get_actuators_by_type(self, actuator_type: str) -> List[str]:
        """
        Get a list of actuator names of a specific type.
        
        Args:
            actuator_type: Type of actuators to find
            
        Returns:
            List of actuator names
        """
        return [name for name, actuator in self.actuators.items() 
                if actuator.type == actuator_type]
