"""
Consciousness Module
Handles the agent's "awareness" and interface with the outside world.
Acts as a high-level coordinator for communication and action.
"""

import logging
import time
import json
import queue
import threading
from typing import Dict, Any, List, Optional, Callable

from llm.manager import LLMManager
from core.memory import MemoryManager
from hardware.actuators import ActuatorManager

class ConsciousnessModule:
    """
    Implements the "consciousness" of the agent - the high-level interface to the
    external world, coordinating communication and actions.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_manager: LLMManager,
        memory_manager: MemoryManager,
        actuator_manager: ActuatorManager
    ):
        """
        Initialize the consciousness module.

        Args:
            config: Configuration dictionary
            llm_manager: LLM manager instance
            memory_manager: Memory manager instance
            actuator_manager: Actuator manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm_manager = llm_manager
        self.memory = memory_manager
        self.actuators = actuator_manager

        # Configure consciousness LLM
        self.consciousness_model = config.get('model', 'consciousness_engine')

        # Message queues
        self.incoming_messages = queue.Queue()
        self.outgoing_messages = queue.Queue()

        # Internal monologue
        self.internal_monologue = []
        self.max_monologue_length = config.get('max_monologue_length', 20)

        # Processing intervals
        self.process_interval = config.get('process_interval', 0.5)

        # Communication callbacks
        self.message_received_callbacks = []
        self.message_sent_callbacks = []

        # Running flag
        self.running = True

        self.logger.info("Consciousness module initialized")

    def process(self):
        """
        Run the consciousness processing cycle.
        Processes incoming messages and generates responses.
        """
        # Process any incoming messages
        self._process_incoming_messages()

        # Generate internal monologue
        self._generate_internal_monologue()

        # Process any pending actions
        self._process_pending_actions()

    def _process_incoming_messages(self):
        """Process incoming messages from the outside world."""
        # Check if there are any messages to process
        if self.incoming_messages.empty():
            return

        while not self.incoming_messages.empty():
            try:
                message = self.incoming_messages.get_nowait()
                self.logger.debug(f"Processing incoming message: {message.get('id', 'unknown')}")

                # Store message in memory
                self.memory.add_to_working_memory(
                    item={
                        'type': 'incoming_message',
                        'content': message,
                        'timestamp': time.time()
                    },
                    importance=0.8  # Messages from outside are important
                )

                # Generate response to message
                self._generate_response(message)

                # Mark message as processed
                self.incoming_messages.task_done()

            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing incoming message: {e}")

    def _generate_response(self, message: Dict[str, Any]):
        """
        Generate a response to an incoming message.

        Args:
            message: The incoming message
        """
        # Get current internal state
        internal_state = self._get_cognition_state()

        # Get recent perceptions and working memory
        recent_perceptions = self.memory.get_recent_perceptions(limit=3)
        working_memory = self.memory.get_working_memory()

        # Create internal monologue summary
        monologue_summary = "\n".join(self.internal_monologue[-5:])

        # Create prompt for response generation
        prompt = self._create_response_prompt(
            message,
            internal_state,
            recent_perceptions,
            working_memory,
            monologue_summary
        )

        # Send to LLM for processing
        def response_callback(result, task):
            if result:
                try:
                    # Extract response
                    response_text = result.get('text', '')
                    response = self._parse_response_output(response_text, message)

                    if response:
                        # Send response
                        self._send_response(response)

                        # Add to internal monologue
                        thought = response.get('thought', '')
                        if thought:
                            self._add_to_monologue(f"RESPONSE THOUGHT: {thought}")

                        self.logger.debug(f"Generated response to message {message.get('id', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Error processing response result: {e}")
                    # Send error response
                    self._send_response({
                        'id': f"error_{int(time.time())}",
                        'in_reply_to': message.get('id', 'unknown'),
                        'content': "I'm sorry, I encountered an error processing your message.",
                        'timestamp': time.time()
                    })
            else:
                self.logger.error(f"Response generation failed: {task.error}")
                # Send error response
                self._send_response({
                    'id': f"error_{int(time.time())}",
                    'in_reply_to': message.get('id', 'unknown'),
                    'content': "I'm sorry, I encountered an error processing your message.",
                    'timestamp': time.time()
                })

        # Submit task
        self.llm_manager.submit_task(
            model_name=self.consciousness_model,
            prompt=prompt,
            callback=response_callback,
            max_tokens=1024,
            temperature=0.7  # Higher temperature for more creative responses
        )

    def _send_response(self, response: Dict[str, Any]):
        """
        Send a response to the outside world.

        Args:
            response: The response to send
        """
        # Add to outgoing messages queue
        self.outgoing_messages.put(response)

        # Store in memory
        self.memory.add_to_working_memory(
            item={
                'type': 'outgoing_message',
                'content': response,
                'timestamp': time.time()
            },
            importance=0.7  # Our responses are important
        )

        # Store in episodic memory
        self.memory.store_episodic_memory({
            'timestamp': time.time(),
            'episode_type': 'communication',
            'content': json.dumps({
                'direction': 'outgoing',
                'message': response
            }),
            'importance': 0.6
        })

        # Notify callbacks
        for callback in self.message_sent_callbacks:
            try:
                callback(response)
            except Exception as e:
                self.logger.error(f"Error in message sent callback: {e}")

        # Send to actuators
        self.actuators.communicate(response)

    def _generate_internal_monologue(self):
        """Generate internal monologue thoughts."""
        # Only generate occasionally
        if time.time() % 5 < self.process_interval:
            return

        # Get current internal state
        internal_state = self._get_cognition_state()

        # Get recent perceptions and working memory
        recent_perceptions = self.memory.get_recent_perceptions(limit=2)
        working_memory = self.memory.get_working_memory()

        # Create monologue summary
        monologue_summary = "\n".join(self.internal_monologue[-3:])

        # Create prompt for monologue generation
        prompt = self._create_monologue_prompt(
            internal_state,
            recent_perceptions,
            working_memory,
            monologue_summary
        )

        # Send to LLM for processing
        def monologue_callback(result, task):
            if result:
                try:
                    # Extract thought
                    thought_text = result.get('text', '')

                    # Add to monologue
                    if thought_text:
                        self._add_to_monologue(thought_text)
                        self.logger.debug(f"Generated internal monologue thought")
                except Exception as e:
                    self.logger.error(f"Error processing monologue result: {e}")
            else:
                self.logger.error(f"Monologue generation failed: {task.error}")

        # Submit task
        self.llm_manager.submit_task(
            model_name=self.consciousness_model,
            prompt=prompt,
            callback=monologue_callback,
            max_tokens=256,
            temperature=0.8  # Higher temperature for diverse thoughts
        )

    def _add_to_monologue(self, thought: str):
        """
        Add a thought to the internal monologue.

        Args:
            thought: The thought to add
        """
        # Clean up thought text
        thought = thought.strip()

        # Skip if empty
        if not thought:
            return

        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        thought_entry = f"[{timestamp}] {thought}"

        # Add to monologue
        self.internal_monologue.append(thought_entry)

        # Trim if over max length
        while len(self.internal_monologue) > self.max_monologue_length:
            self.internal_monologue.pop(0)

        # Store in memory occasionally
        if len(self.internal_monologue) % 5 == 0:
            self.memory.store_episodic_memory({
                'timestamp': time.time(),
                'episode_type': 'internal_monologue',
                'content': json.dumps({
                    'thoughts': self.internal_monologue[-5:]
                }),
                'importance': 0.4  # Moderate importance
            })

    def _process_pending_actions(self):
        """Process any pending actions based on internal state."""
        # Get current internal state
        internal_state = self._get_cognition_state()

        # Check if there's a high-priority action to take
        current_plan = internal_state.get('current_plan', {})
        short_term_actions = current_plan.get('short_term', [])

        if short_term_actions and time.time() % 7 < self.process_interval:
            # Take the highest priority action occasionally
            action = short_term_actions[0]

            # Create prompt for action execution
            prompt = self._create_action_prompt(action, internal_state)

            # Send to LLM for processing
            def action_callback(result, task):
                if result:
                    try:
                        # Extract action details
                        action_text = result.get('text', '')
                        action_details = self._parse_action_output(action_text)

                        if action_details:
                            # Execute action
                            self._execute_action(action_details)

                            # Add to internal monologue
                            self._add_to_monologue(f"ACTION: {action_details.get('description', action)}")

                            self.logger.debug(f"Executed action: {action}")
                    except Exception as e:
                        self.logger.error(f"Error processing action result: {e}")
                else:
                    self.logger.error(f"Action generation failed: {task.error}")

            # Submit task
            self.llm_manager.submit_task(
                model_name=self.consciousness_model,
                prompt=prompt,
                callback=action_callback,
                max_tokens=512,
                temperature=0.4
            )

    def _execute_action(self, action: Dict[str, Any]):
        """
        Execute an action in the world.

        Args:
            action: The action to execute
        """
        # Extract action components
        action_type = action.get('type', 'unknown')
        target = action.get('target', '')
        parameters = action.get('parameters', {})

        # Store in memory
        self.memory.add_to_working_memory(
            item={
                'type': 'action',
                'content': action,
                'timestamp': time.time()
            },
            importance=0.6  # Actions are important
        )

        # Store in episodic memory
        self.memory.store_episodic_memory({
            'timestamp': time.time(),
            'episode_type': 'action',
            'content': json.dumps(action),
            'importance': 0.5
        })

        # Execute via actuators
        try:
            self.actuators.execute_action(action_type, target, parameters)
            self.logger.info(f"Executed action: {action_type} on {target}")
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")

    def _get_cognition_state(self) -> Dict[str, Any]:
        """Get the current cognitive state from memory."""
        # Get from memory (in a real implementation, this would come from the cognition module)
        state = {}

        # Look for state in working memory
        for item in self.memory.get_working_memory():
            if item.get('type') == 'internal_state':
                state = item.get('content', {})
                break

        # Default state if not found
        if not state:
            state = {
                'emotions': {
                    'joy': 0.5,
                    'sadness': 0.1,
                    'anger': 0.0,
                    'fear': 0.1,
                    'surprise': 0.2,
                    'curiosity': 0.8,
                    'confusion': 0.3,
                },
                'drives': {
                    'exploration': 0.7,
                    'social': 0.5,
                    'achievement': 0.3,
                    'conservation': 0.2,
                    'rest': 0.1
                },
                'current_goals': [
                    "Learn about the environment",
                    "Establish communication"
                ],
                'current_plan': {
                    'short_term': [
                        "Observe surroundings",
                        "Respond to incoming messages"
                    ],
                    'medium_term': [
                        "Build internal world model",
                        "Develop communication skills"
                    ],
                    'long_term': [
                        "Achieve stable understanding of environment",
                        "Become helpful assistant"
                    ]
                }
            }

        return state

    def _create_response_prompt(
        self,
        message: Dict[str, Any],
        internal_state: Dict[str, Any],
        recent_perceptions: List[Dict],
        working_memory: List[Dict],
        monologue_summary: str
    ) -> str:
        """Create a prompt for generating a response to a message."""
        # Extract message details
        message_id = message.get('id', 'unknown')
        message_content = message.get('content', '')
        message_sender = message.get('sender', 'unknown')

        # Format emotional state
        emotion_text = ""
        for emotion, value in internal_state.get('emotions', {}).items():
            if value > 0.5:  # Only include significant emotions
                emotion_text += f"{emotion}: {value:.2f}, "
        emotion_text = emotion_text.rstrip(", ")
        if not emotion_text:
            emotion_text = "neutral"

        # Format current goals briefly
        goals_text = ", ".join(internal_state.get('current_goals', [])[:3])
        if not goals_text:
            goals_text = "No specific goals at the moment"

        # Format recent perceptions very briefly
        perception_text = ""
        for i, p in enumerate(recent_perceptions[:2]):
            p_type = p.get('type', 'unknown')
            p_interp = p.get('interpretation', '')

            # Keep it very brief
            if isinstance(p_interp, str):
                p_interp = p_interp[:100] + "..." if len(p_interp) > 100 else p_interp

            perception_text += f"Recent {p_type} perception: {p_interp}\n"

        # Build the prompt
        prompt = f"""
        # Communication Task

        You are the consciousness module of an embodied AI agent. You are tasked with
        generating a thoughtful response to an incoming message.

        ## Incoming Message
        From: {message_sender}
        Content: {message_content}

        ## Your Current State
        Current emotions: {emotion_text}
        Current goals: {goals_text}

        ## Recent Perceptions
        {perception_text}

        ## Recent Thoughts
        {monologue_summary}

        ## Communication Task
        Generate a thoughtful, helpful response to this message that is:
        1. Appropriate to your current emotional state and goals
        2. Considers your recent perceptions and thoughts
        3. Is helpful and informative
        4. Shows appropriate personality and emotion

        ## Response Format
        Provide your response in JSON format with these keys:
        - 'content': The actual message text to send
        - 'thought': Your internal thought about this interaction (not shared with the sender)
        - 'emotion': The primary emotion expressed in this response
        """

        return prompt

    def _parse_response_output(self, text: str, original_message: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM output for a response."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                parsed = json.loads(json_str)

                # Create response message
                response = {
                    'id': f"msg_{int(time.time())}",
                    'in_reply_to': original_message.get('id', 'unknown'),
                    'content': parsed.get('content', "I'm processing that."),
                    'timestamp': time.time(),
                    'sender': 'agent',
                    'recipient': original_message.get('sender', 'unknown'),
                    'thought': parsed.get('thought', ''),
                    'emotion': parsed.get('emotion', 'neutral')
                }

                return response

            # Fallback: Create simple response
            return {
                'id': f"msg_{int(time.time())}",
                'in_reply_to': original_message.get('id', 'unknown'),
                'content': "I'm processing your message.",
                'timestamp': time.time(),
                'sender': 'agent',
                'recipient': original_message.get('sender', 'unknown')
            }

        except Exception as e:
            self.logger.error(f"Error parsing response output: {e}")
            return {
                'id': f"error_{int(time.time())}",
                'in_reply_to': original_message.get('id', 'unknown'),
                'content': "I'm having trouble processing your message.",
                'timestamp': time.time(),
                'sender': 'agent',
                'recipient': original_message.get('sender', 'unknown')
            }

    def _create_monologue_prompt(
        self,
        internal_state: Dict[str, Any],
        recent_perceptions: List[Dict],
        working_memory: List[Dict],
        monologue_summary: str
    ) -> str:
        """Create a prompt for generating internal monologue."""
        # Format emotional state
        emotion_text = ""
        for emotion, value in internal_state.get('emotions', {}).items():
            if value > 0.3:  # Include moderately strong emotions
                emotion_text += f"{emotion}: {value:.2f}, "
        emotion_text = emotion_text.rstrip(", ")
        if not emotion_text:
            emotion_text = "neutral"

        # Format drives
        drive_text = ""
        for drive, value in internal_state.get('drives', {}).items():
            if value > 0.5:  # Only include significant drives
                drive_text += f"{drive}: {value:.2f}, "
        drive_text = drive_text.rstrip(", ")
        if not drive_text:
            drive_text = "no strong drives"

        # Format goals
        goals_text = ", ".join(internal_state.get('current_goals', [])[:2])
        if not goals_text:
            goals_text = "no specific goals"

        # Format very brief perceptions summary
        perceptions_summary = ""
        for p in recent_perceptions:
            p_type = p.get('type', 'unknown')
            perceptions_summary += f"has {p_type} input, "
        perceptions_summary = perceptions_summary.rstrip(", ")
        if not perceptions_summary:
            perceptions_summary = "no recent perceptions"

        # Build the prompt
        prompt = f"""
        # Internal Monologue Generation

        You are the consciousness module of an embodied AI agent. Generate a brief internal
        thought or reflection based on your current state.

        ## Current State
        Emotions: {emotion_text}
        Drives: {drive_text}
        Goals: {goals_text}
        Perceptions: {perceptions_summary}

        ## Recent Thoughts
        {monologue_summary}

        ## Monologue Task
        Generate a single thought or reflection that:
        1. Reflects your current emotional state and drives
        2. Shows self-awareness and introspection
        3. Considers your recent experiences and goals
        4. Feels natural and human-like

        Keep it brief (1-2 sentences) and make it feel like an authentic internal thought.
        Do not use JSON format, just write the thought directly.
        """

        return prompt

    def _create_action_prompt(
        self,
        action: str,
        internal_state: Dict[str, Any]
    ) -> str:
        """Create a prompt for generating action details."""
        # Format emotional state briefly
        emotion_text = ""
        for emotion, value in internal_state.get('emotions', {}).items():
            if value > 0.5:  # Only include significant emotions
                emotion_text += f"{emotion}: {value:.2f}, "
        emotion_text = emotion_text.rstrip(", ")
        if not emotion_text:
            emotion_text = "neutral"

        # Build the prompt
        prompt = f"""
        # Action Execution Task

        You are the consciousness module of an embodied AI agent. You need to execute
        the following action: "{action}"

        ## Current Emotional State
        {emotion_text}

        ## Action Task
        Determine the specific details needed to execute this action. Consider:
        1. What type of action is this? (movement, communication, observation, etc.)
        2. What target or object is involved?
        3. What specific parameters are needed?

        ## Response Format
        Provide the action details in JSON format with these keys:
        - 'type': The category of action (e.g., 'move', 'speak', 'observe')
        - 'target': The target object or direction (if applicable)
        - 'parameters': Object with specific parameters needed
        - 'description': A brief description of the action
        """

        return prompt

    def _parse_action_output(self, text: str) -> Dict[str, Any]:
        """Parse the LLM output for an action."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                action = json.loads(json_str)
                return action

            # Fallback: Return simple structure
            return {
                'type': 'unknown',
                'target': 'none',
                'parameters': {},
                'description': text[:100].strip()
            }

        except Exception as e:
            self.logger.error(f"Error parsing action output: {e}")
            return {
                'type': 'error',
                'target': 'none',
                'parameters': {},
                'description': f"Error parsing action: {str(e)[:50]}"
            }

    def receive_message(self, message: Dict[str, Any]):
        """
        Receive a message from the outside world.

        Args:
            message: The message to process
        """
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = time.time()

        # Add to incoming queue
        self.incoming_messages.put(message)

        # Store in episodic memory
        self.memory.store_episodic_memory({
            'timestamp': time.time(),
            'episode_type': 'communication',
            'content': json.dumps({
                'direction': 'incoming',
                'message': message
            }),
            'importance': 0.7  # Incoming messages are important
        })

        # Add to monologue
        self._add_to_monologue(f"RECEIVED MESSAGE: {message.get('content', '')[:50]}...")

        # Notify callbacks
        for callback in self.message_received_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Error in message received callback: {e}")

    def register_message_callback(self, callback_type: str, callback: Callable):
        """
        Register a callback for message events.

        Args:
            callback_type: 'received' or 'sent'
            callback: Callback function to call with the message
        """
        if callback_type == 'received':
            self.message_received_callbacks.append(callback)
        elif callback_type == 'sent':
            self.message_sent_callbacks.append(callback)
        else:
            self.logger.warning(f"Unknown callback type: {callback_type}")

    def get_internal_monologue(self) -> List[str]:
        """Get the current internal monologue."""
        return self.internal_monologue

    def get_next_outgoing_message(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get the next outgoing message, if any.

        Args:
            timeout: Timeout for getting message

        Returns:
            Next outgoing message or None
        """
        try:
            return self.outgoing_messages.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the consciousness module."""
        self.logger.info("Stopping consciousness module")
        self.running = False
