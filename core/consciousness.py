"""
Consciousness Module with improved error handling
Handles the agent's "awareness" and interface with the outside world.
Acts as a high-level coordinator for communication and action.
"""

import logging
import time
import json
import queue
import threading
import traceback
from typing import Dict, Any, List, Optional, Callable

class ConsciousnessModule:
    """
    Implements the "consciousness" of the agent - the high-level interface to the
    external world, coordinating communication and actions.
    """

    def __init__(
        self,
        config,
        llm_manager,
        memory_manager,
        actuator_manager
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
        try:
            # Process any incoming messages
            self._process_incoming_messages()

            # Generate internal monologue
            self._generate_internal_monologue()

            # Process any pending actions
            self._process_pending_actions()
        except Exception as e:
            self.logger.error(f"Error in consciousness processing: {e}")
            self.logger.error(traceback.format_exc())


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

                # Add to monologue
                self._add_to_monologue(f"RECEIVED MESSAGE: {message.get('content', '')[:50]}...")

                # Generate response to message - with more error handling
                try:
                    self._generate_response(message)
                except Exception as e:
                    self.logger.error(f"Error in generate_response: {e}")
                    self.logger.error(traceback.format_exc())

                    # If there's an error generating a response, create a simple one now
                    simple_response = {
                        'id': f"simple_{int(time.time())}",
                        'in_reply_to': message.get('id', 'unknown'),
                        'content': "I've received your message and am processing it. My response system is experiencing some delays.",
                        'timestamp': time.time(),
                        'sender': 'agent',
                        'recipient': message.get('sender', 'unknown'),
                        'thought': "Internal error occurred, providing simple response"
                    }

                    # Send the simple response directly
                    self._send_response(simple_response)

                # Mark message as processed
                self.incoming_messages.task_done()

            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing incoming message: {e}")
                self.logger.error(traceback.format_exc())


    def _generate_response(self, message):
        """
        Generate a response to an incoming message.

        Args:
            message: The incoming message
        """
        try:
            # Verify the model exists
            if self.consciousness_model not in self.llm_manager.models:
                self.logger.error(f"Model {self.consciousness_model} not found in LLM manager!")
                # Fallback to a direct response using any available model
                model_names = list(self.llm_manager.models.keys())
                if model_names:
                    self.consciousness_model = model_names[0]
                    self.logger.info(f"Falling back to model {self.consciousness_model}")
                else:
                    raise Exception("No LLM models available")

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

            # Debug log the prompt
            self.logger.debug(f"Response prompt for message {message.get('id', 'unknown')}:\n{prompt[:500]}...")

            # Create callback function for handling response
            def response_callback(result, task):
                try:
                    self.logger.debug(f"LLM response callback triggered for {message.get('id', 'unknown')}")

                    if result:
                        # Extract response
                        response_text = result.get('text', '')
                        self.logger.debug(f"Received response from LLM: {response_text[:100]}...")

                        # Parse the response
                        response = self._parse_response_output(response_text, message)

                        if response:
                            # Send response
                            self._send_response(response)

                            # Add to internal monologue
                            thought = response.get('thought', '')
                            if thought:
                                self._add_to_monologue(f"RESPONSE THOUGHT: {thought}")

                            self.logger.debug(f"Generated response to message {message.get('id', 'unknown')}")
                        else:
                            # If parsing failed, send a fallback
                            self.logger.warning(f"Failed to parse response for message {message.get('id', 'unknown')}")
                            fallback_response = self._generate_fallback_response(message)
                            self._send_response(fallback_response)
                    else:
                        # Generate and send fallback response
                        self.logger.warning(f"No result from LLM for message {message.get('id', 'unknown')}")
                        fallback_response = self._generate_fallback_response(message)
                        self._send_response(fallback_response)

                except Exception as e:
                    self.logger.error(f"Error in response callback: {e}")
                    self.logger.error(traceback.format_exc())

                    # Send error response
                    fallback_response = self._generate_fallback_response(message)
                    self._send_response(fallback_response)

            # Submit task with a timeout
            self.logger.info(f"Sending message to LLM: {message.get('id', 'unknown')}")

            # Try both async submission and direct method if async fails
            try:
                task_id = self.llm_manager.submit_task(
                    model_name=self.consciousness_model,
                    prompt=prompt,
                    callback=response_callback,
                    max_tokens=1024,
                    temperature=0.7
                )
                self.logger.debug(f"Task submitted with ID: {task_id}")
            except Exception as e:
                self.logger.error(f"Error submitting LLM task: {e}")

                # Try direct synchronous method instead
                self.logger.info("Trying direct LLM response as fallback")
                if not self.direct_llm_response(message):
                    # If that also fails, send a simple response
                    fallback_response = self._generate_fallback_response(message)
                    self._send_response(fallback_response)

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self.logger.error(traceback.format_exc())

            # Send error response
            fallback_response = self._generate_fallback_response(message)
            self._send_response(fallback_response)

    def _generate_fallback_response(self, message):
        """Generate a fallback response when the LLM fails to respond."""
        self.logger.warning(f"Generating fallback response for message {message.get('id', 'unknown')}")

        # Create a simple fallback response
        return {
            'id': f"fallback_{int(time.time())}",
            'in_reply_to': message.get('id', 'unknown'),
            'content': "I'm sorry, I'm having trouble processing your message right now. My language processing system is experiencing a delay. Please try again or rephrase your query.",
            'timestamp': time.time(),
            'sender': 'agent',
            'recipient': message.get('sender', 'unknown'),
            'thought': "Internal systems are delayed in processing this message. Providing a fallback response while the issue is resolved.",
            'fallback': True
        }


    def _send_response(self, response):
        """
        Send a response to the outside world.

        Args:
            response: The response to send
        """
        try:
            # Add to outgoing messages queue
            self.logger.debug(f"Adding response to outgoing queue: {response.get('id', 'unknown')}")
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
            self.logger.debug(f"Notifying {len(self.message_sent_callbacks)} message sent callbacks")
            for callback in self.message_sent_callbacks:
                try:
                    callback(response)
                except Exception as e:
                    self.logger.error(f"Error in message sent callback: {e}")

            # Send to actuators
            self.logger.debug(f"Sending to actuators: {response.get('id', 'unknown')}")
            result = self.actuators.communicate(response)
            if not result:
                self.logger.warning(f"Failed to communicate via actuators: {response.get('id', 'unknown')}")

            self.logger.info(f"Sent response: {response.get('content', '')[:100]}...")

        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            self.logger.error(traceback.format_exc())

    def _generate_internal_monologue(self):
        """Generate internal monologue thoughts."""
        try:
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

            # Create callback for monologue generation
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
                    self.logger.debug(f"Monologue generation returned no result")

            # Submit task
            self.llm_manager.submit_task(
                model_name=self.consciousness_model,
                prompt=prompt,
                callback=monologue_callback,
                max_tokens=256,
                temperature=0.8  # Higher temperature for diverse thoughts
            )

        except Exception as e:
            self.logger.error(f"Error generating internal monologue: {e}")
            self.logger.error(traceback.format_exc())

    def _add_to_monologue(self, thought):
        """
        Add a thought to the internal monologue.

        Args:
            thought: The thought to add
        """
        try:
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

            self.logger.debug(f"Added to monologue: {thought[:50]}...")

        except Exception as e:
            self.logger.error(f"Error adding to monologue: {e}")
            self.logger.error(traceback.format_exc())

    def _process_pending_actions(self):
        """Process any pending actions based on internal state."""
        try:
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

                # Create callback for action execution
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
                        self.logger.debug(f"Action generation returned no result")

                # Submit task
                self.llm_manager.submit_task(
                    model_name=self.consciousness_model,
                    prompt=prompt,
                    callback=action_callback,
                    max_tokens=512,
                    temperature=0.4
                )

        except Exception as e:
            self.logger.error(f"Error processing pending actions: {e}")
            self.logger.error(traceback.format_exc())

    def _execute_action(self, action):
        """
        Execute an action in the world.

        Args:
            action: The action to execute
        """
        try:
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
            self.actuators.execute_action(action_type, target, parameters)
            self.logger.info(f"Executed action: {action_type} on {target}")

        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            self.logger.error(traceback.format_exc())

    def _get_cognition_state(self):
        """Get the current cognitive state from memory."""
        try:
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

        except Exception as e:
            self.logger.error(f"Error getting cognition state: {e}")
            self.logger.error(traceback.format_exc())

            # Return empty state on error
            return {}


    def _create_response_prompt(self, message, internal_state, recent_perceptions, working_memory, monologue_summary):
        """Create a prompt for generating a response to a message."""
        try:
            # Extract message details
            message_content = message.get('content', '')
            message_sender = message.get('sender', 'unknown')

            # Format recent perceptions
            perception_text = ""
            for i, p in enumerate(recent_perceptions[:2]):
                p_type = p.get('type', 'unknown')
                p_interp = p.get('interpretation', '')

                if p_interp:
                    perception_text += f"Recent {p_type}: {p_interp[:100]}...\n"

            if not perception_text:
                perception_text = "No recent perceptions available."

            # Try to retrieve episodic memories about this person
            try:
                sender_memories = self.memory.retrieve_episodic_memories(
                    query=message_sender,
                    limit=3
                )

                memory_text = ""
                for i, m in enumerate(sender_memories):
                    m_type = m.get('episode_type', 'unknown')
                    m_content = m.get('content', {})

                    try:
                        if isinstance(m_content, str):
                            m_content = json.loads(m_content)

                        if m_type == 'communication':
                            m_msg = m_content.get('message', {})
                            m_content_str = m_msg.get('content', '')[:100]
                            m_dir = m_content.get('direction', 'unknown')
                            memory_text += f"Past {m_dir} message: {m_content_str}...\n"
                    except:
                        memory_text += f"Memory: {str(m_content)[:100]}...\n"

                if not memory_text:
                    memory_text = "No memories of this person available."
            except:
                memory_text = "Memory retrieval error."

            # Format current state
            emotion_text = ""
            for emotion, value in internal_state.get('emotions', {}).items():
                if value > 0.3:
                    emotion_text += f"{emotion}: {value:.2f}, "
            emotion_text = emotion_text.rstrip(", ")
            if not emotion_text:
                emotion_text = "neutral"

            # Build the prompt
            prompt = f"""
    # Communication Task

    You are an embodied AI assistant named Lumina with access to cameras and microphones.
    You have received the following message:

    Message: {message_content}

    ## Recent Perceptions
    {perception_text}

    ## Memories About This Person
    {memory_text}

    ## Internal State
    Emotion: {emotion_text}
    Goals: {', '.join(internal_state.get('current_goals', ['Learn about the environment', 'Communicate effectively'])[:2])}

    ## Response Task
    Generate a thoughtful, helpful response that:
    1. Addresses the user's message directly
    2. References your perceptions when relevant
    3. References your memories about this person when relevant
    4. Shows appropriate personality and emotion
    5. Is friendly and engaging

    ## Response Format
    Provide your response in JSON format with these keys:
    - 'content': The actual message text to send
    - 'thought': Your internal thought about this interaction (not shared with the sender)
    """

            return prompt

        except Exception as e:
            self.logger.error(f"Error creating response prompt: {e}")
            self.logger.error(traceback.format_exc())

            # Return simple prompt on error
            return f"Generate a response to this message: {message.get('content', '')}"


    def _parse_response_output(self, text, original_message):
        """Parse the LLM output for a response."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                # Extract the JSON portion
                json_str = text[start_idx:end_idx+1]

                try:
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

                    self.logger.info(f"Successfully parsed response: {response['content'][:50]}...")
                    return response
                except json.JSONDecodeError as json_err:
                    self.logger.error(f"JSON parse error: {json_err}. JSON string: {json_str[:100]}...")

            # Could not find valid JSON, try to extract content directly
            self.logger.warning(f"Could not parse JSON from response, creating simple response")

            # Try to extract content between content markers if available
            content_start = text.find("content")
            if content_start > 0:
                content_start = text.find(":", content_start) + 1
                content_end = text.find(",", content_start)
                if content_end < 0:
                    content_end = text.find("}", content_start)

                if content_start > 0 and content_end > content_start:
                    content = text[content_start:content_end].strip().strip('"\'')
                    if content:
                        return {
                            'id': f"msg_{int(time.time())}",
                            'in_reply_to': original_message.get('id', 'unknown'),
                            'content': content,
                            'timestamp': time.time(),
                            'sender': 'agent',
                            'recipient': original_message.get('sender', 'unknown')
                        }

            # Fallback to using the whole text if it's not too long
            if len(text) < 200:
                return {
                    'id': f"msg_{int(time.time())}",
                    'in_reply_to': original_message.get('id', 'unknown'),
                    'content': text.strip(),
                    'timestamp': time.time(),
                    'sender': 'agent',
                    'recipient': original_message.get('sender', 'unknown')
                }

            # Final fallback
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
            self.logger.error(traceback.format_exc())
            return {
                'id': f"msg_{int(time.time())}",
                'in_reply_to': original_message.get('id', 'unknown'),
                'content': "I'm experiencing some internal processing issues.",
                'timestamp': time.time(),
                'sender': 'agent',
                'recipient': original_message.get('sender', 'unknown')
            }

    def _create_monologue_prompt(self, internal_state, recent_perceptions, working_memory, monologue_summary):
        """Create a prompt for generating internal monologue."""
        try:
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

        except Exception as e:
            self.logger.error(f"Error creating monologue prompt: {e}")
            self.logger.error(traceback.format_exc())

            # Return simple prompt on error
            return "Generate a brief internal thought for an AI agent."

    def _create_action_prompt(self, action, internal_state):
        """Create a prompt for generating action details."""
        try:
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

        except Exception as e:
            self.logger.error(f"Error creating action prompt: {e}")
            self.logger.error(traceback.format_exc())

            # Return simple prompt on error
            return f"Generate action details for this action: {action}"

    def _parse_action_output(self, text):
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
            self.logger.warning(f"Could not parse JSON from action output: {text[:100]}...")
            return {
                'type': 'unknown',
                'target': 'none',
                'parameters': {},
                'description': text[:100].strip()
            }

        except Exception as e:
            self.logger.error(f"Error parsing action output: {e}")
            self.logger.error(traceback.format_exc())
            return None

    # Update in consciousness.py
    def receive_message(self, message):
        """
        Receive a message from the outside world.

        Args:
            message: The message to process
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = time.time()

            # Add to incoming queue
            self.incoming_messages.put(message)

            # Debug message flow
            self.debug_message_flow(message.get('id', 'unknown'))

            # Immediately trigger processing instead of waiting for the loop
            self._process_incoming_messages()

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

            # Notify callbacks
            for callback in self.message_received_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Error in message received callback: {e}")

            self.logger.info(f"Received message: {message.get('content', '')[:100]}...")

        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            self.logger.error(traceback.format_exc())

    def register_message_callback(self, callback_type, callback):
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

    def get_internal_monologue(self):
        """Get the current internal monologue."""
        return self.internal_monologue

    def get_next_outgoing_message(self, timeout=0.1):
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


    def direct_llm_response(self, message):
        """
        Generate a response directly using the LLM without complex processing.

        Args:
            message: The incoming message
        """
        self.logger.info(f"Generating direct LLM response for {message.get('id', 'unknown')}")

        # Create a simple prompt
        prompt = f"""
        You are an AI assistant. Please respond helpfully to this message:

        {message.get('content', '')}

        Respond in a friendly, helpful manner.
        """

        try:
            # Submit synchronous request to LLM with a timeout
            result = self.llm_manager.submit_task_sync(
                model_name=self.consciousness_model,
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                timeout=10.0  # 10 second timeout
            )

            if result and 'text' in result:
                # Create response
                response = {
                    'id': f"direct_{int(time.time())}",
                    'in_reply_to': message.get('id', 'unknown'),
                    'content': result['text'],
                    'timestamp': time.time(),
                    'sender': 'agent',
                    'recipient': message.get('sender', 'unknown'),
                    'thought': "Direct LLM response due to processing issues"
                }

                # Send the response
                self._send_response(response)
                return True

        except Exception as e:
            self.logger.error(f"Error in direct LLM response: {e}")

        return False

    def debug_message_flow(self, message_id):
        """Print the current state of message processing for debugging."""
        self.logger.info(f"DEBUG MESSAGE FLOW for {message_id}")
        self.logger.info(f"- Incoming queue size: {self.incoming_messages.qsize()}")
        self.logger.info(f"- Outgoing queue size: {self.outgoing_messages.qsize()}")
        self.logger.info(f"- Callbacks registered: received={len(self.message_received_callbacks)}, sent={len(self.message_sent_callbacks)}")
        self.logger.info(f"- Internal monologue length: {len(self.internal_monologue)}")

        # Check if consciousness model is configured correctly
        self.logger.info(f"- Using consciousness model: {self.consciousness_model}")
        if self.consciousness_model not in self.llm_manager.models:
            self.logger.error(f"Model {self.consciousness_model} not found in LLM manager!")
        else:
            self.logger.info(f"- Model {self.consciousness_model} is available")

    def stop(self):
        """Stop the consciousness module."""
        self.logger.info("Stopping consciousness module")
        self.running = False
