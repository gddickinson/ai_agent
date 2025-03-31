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
import re

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
            self.outgoing_messages.put(response)

            # Store in conversation history
            self.memory.store_conversation_chunk('agent', response.get('content', ''))

            # Store in working memory
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

            # Check if we should summarize the conversation
            # Get count of recent messages
            recent_conversation = self.memory.get_recent_conversation(limit=20)
            if len(recent_conversation) >= 10:  # After 10 exchanges, create a summary
                # Check when the last summary was created
                last_summary_time = self.memory.get_last_summary_time()
                current_time = time.time()

                # If no summary in the last 5 minutes, create one
                if not last_summary_time or (current_time - last_summary_time) > 300:
                    summary = self._summarize_and_store_conversation()
                    if summary:
                        self.logger.info(f"Created conversation summary: {summary[:100]}...")

            # Notify callbacks
            for callback in self.message_sent_callbacks:
                try:
                    callback(response)
                except Exception as e:
                    self.logger.error(f"Error in message sent callback: {e}")

            # Send to actuators
            self.actuators.communicate(response)

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

            # Get recent conversation
            recent_conversation = self.memory.get_recent_conversation(limit=5)
            conversation_text = ""
            for item in recent_conversation:
                speaker = "User" if item['speaker'] != 'agent' else "You"
                conversation_text += f"{speaker}: {item['content']}\n"

            # Check if we know the user
            user_info = self.memory.get_user_info(user_id=message_sender)
            user_name = user_info.get('name', 'unknown') if user_info else 'unknown'

            # Get known facts about the user
            user_facts = []
            if user_name != 'unknown':
                facts = self.memory.get_facts(entity=f"user:{message_sender}")
                if facts:
                    for fact in facts:
                        user_facts.append(f"{fact['attribute']}: {fact['value']}")

            facts_text = "\n".join(user_facts) if user_facts else "No specific facts known about this user."

            # Get relevant conversation summaries
            previous_conversations = ""
            if user_name != 'unknown':
                summaries = self.memory.get_conversation_summaries(user_id=message_sender, limit=3)
                if summaries:
                    previous_conversations = "\nPrevious conversations:\n"
                    for summary in summaries:
                        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(summary['timestamp']))
                        previous_conversations += f"- {timestamp}: {summary['summary']}\n"

            # Identify topics and get related memories
            topics = self._identify_conversation_topics(message_content)
            related_memories = []

            if topics:
                for topic in topics:
                    memories = self.memory.get_related_memories(topic, limit=2)
                    related_memories.extend(memories)

            # Format related memories
            topic_memories = ""
            if related_memories:
                topic_memories = "\nRelated memories:\n"
                for memory in related_memories:
                    if memory['type'] == 'conversation_summary':
                        timestamp = time.strftime("%Y-%m-%d", time.localtime(memory['timestamp']))
                        topic_memories += f"- {timestamp}: {memory['content']}\n"
                    elif memory['type'] == 'fact':
                        # Format facts
                        topic_memories += f"- I know that {memory['attribute']}: {memory['value']}\n"

            # Format recent perceptions
            perceptions = self.memory.get_recent_perceptions(limit=3)
            perception_text = ""
            for p in perceptions:
                if p.get('type') == 'camera':
                    interp = p.get('interpretation', '')
                    if isinstance(interp, str) and len(interp) > 0:
                        try:
                            interp_data = json.loads(interp)
                            perception_text += f"I can see: {interp_data.get('description', '')}\n"
                        except:
                            perception_text += f"I can see: {interp[:100]}\n"

            if not perception_text:
                perception_text = "No visual information available."

            # Check for specific questions about known facts
            # This helps the agent respond confidently to direct questions
            question_type = "general"

            # Check if the message is asking about age
            if re.search(r'(?:how old|what.*age|age.*what)', message_content.lower()):
                question_type = "age"

            # Build response prompt with all available context
            prompt = f"""
    # Communication Task

    You are Lumina, an embodied AI assistant. You need to respond to this message from {user_name if user_name != 'unknown' else 'a user'}:

    Message: {message_content}

    ## Recent Conversation
    {conversation_text}

    ## User Information
    {"Name: " + user_name if user_name != 'unknown' else "I don't know this user's name yet"}

    ## Known Facts About User
    {facts_text}

    ## Previous Interactions
    {previous_conversations}

    ## Topic-Related Memories
    {topic_memories}

    ## Visual Perception
    {perception_text}

    ## Response Task
    Generate a response that:
    1. Directly addresses the user's message
    2. Uses information from recent conversation and known facts when relevant
    3. References previous conversations when appropriate
    4. Incorporates topic-related memories naturally
    5. Shows personality and appropriate emotion
    6. Is helpful and informative
    7. Is concise and to the point

    ## Special Instructions
    {f"The user is asking about their age. If you know their age, confidently state it in your response." if question_type == "age" else ""}

    ## Format Guidelines
    You MUST format your response as a valid JSON object with these exact keys:
    - "content": Your response message text (this is what will be sent to the user)
    - "thought": Your private thought about this interaction (not shared with the user)

    Format your response exactly like this:
    {{
      "content": "Your response goes here",
      "thought": "Your private thought goes here"
    }}
    """

            self.logger.debug(f"Generated prompt with user facts: {facts_text}")
            return prompt

        except Exception as e:
            self.logger.error(f"Error creating response prompt: {e}")
            self.logger.error(traceback.format_exc())

            # Return simple prompt on error
            return f"Generate a response to this message: {message.get('content', '')}"


    def _parse_response_output(self, text, original_message):
        """Parse the LLM output for a response."""
        try:
            # Detect if response is already pure JSON
            if text.strip().startswith('{') and text.strip().endswith('}'):
                try:
                    parsed = json.loads(text)
                    if 'content' in parsed:
                        # Create response message with just the content
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
                except json.JSONDecodeError:
                    # Fall through to regular parsing
                    pass

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

                    # Try to extract content using regex
                    content_match = re.search(r'"content"\s*:\s*"([^"]+)"', json_str)
                    if content_match:
                        content = content_match.group(1)
                        return {
                            'id': f"msg_{int(time.time())}",
                            'in_reply_to': original_message.get('id', 'unknown'),
                            'content': content,
                            'timestamp': time.time(),
                            'sender': 'agent',
                            'recipient': original_message.get('sender', 'unknown')
                        }

            # If we get here, either no JSON-like structure was found or parsing failed

            # Look for first paragraph that seems like a response
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                # Skip if it looks like code or metadata
                if paragraph.strip().startswith('```') or ':' in paragraph and len(paragraph.strip()) < 50:
                    continue

                # Return the first substantial paragraph
                if len(paragraph.strip()) > 10:
                    return {
                        'id': f"msg_{int(time.time())}",
                        'in_reply_to': original_message.get('id', 'unknown'),
                        'content': paragraph.strip(),
                        'timestamp': time.time(),
                        'sender': 'agent',
                        'recipient': original_message.get('sender', 'unknown')
                    }

            # If all else fails, just use the entire text
            return {
                'id': f"msg_{int(time.time())}",
                'in_reply_to': original_message.get('id', 'unknown'),
                'content': text.strip(),
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

            # Track user information
            sender = message.get('sender', 'unknown')
            content = message.get('content', '')

            # Check if message contains direct age statement
            if "my age is" in content.lower():
                age_match = re.search(r'my age is (\d+)', content.lower())
                if age_match:
                    age = age_match.group(1)
                    self.logger.info(f"Storing explicit age statement: {age}")
                    self.memory.store_fact(f"user:{sender}", "age", age)

            # Store conversation chunk
            self.memory.store_conversation_chunk(sender, content)

            # Extract facts
            self._extract_facts(message)

            # Check if a name is mentioned in the message
            name_match = re.search(r'my name is (\w+)', content.lower())
            if name_match:
                user_name = name_match.group(1).capitalize()
                self.logger.info(f"Detected user name: {user_name}")

                # Store user info
                self.memory.store_user_info(sender, user_name)
            else:
                # Check if we have a name from previous messages
                user_info = self.memory.get_user_info(user_id=sender)
                if not user_info:
                    # Check if name is mentioned in the message directly
                    # Common names detection
                    common_names = ["George", "John", "Mary", "David", "Sarah", "Michael", "Emma", "James",
                                   "Alice", "Bob", "Peter", "Susan", "Tom", "Linda", "Richard", "Emily"]

                    for name in common_names:
                        if name.lower() in content.lower():
                            self.logger.info(f"Detected potential user name in message: {name}")
                            self.memory.store_user_info(sender, name)
                            break

            # Add to incoming queue
            self.incoming_messages.put(message)

            # Debug message flow
            self.debug_message_flow(message.get('id', 'unknown'))

            # Immediately trigger processing instead of waiting for the loop
            self._process_incoming_messages()

            # Store in episodic memory
            try:
                self.memory.store_episodic_memory({
                    'timestamp': time.time(),
                    'episode_type': 'communication',
                    'content': json.dumps({
                        'direction': 'incoming',
                        'message': message
                    }),
                    'importance': 0.7  # Incoming messages are important
                })
            except Exception as e:
                self.logger.error(f"Error storing communication in episodic memory: {e}")

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


    # Update in consciousness.py, in the _extract_facts method
    def _extract_facts(self, message):
        """Extract important facts from a message."""
        try:
            sender = message.get('sender', 'unknown')
            content = message.get('content', '')

            # Expanded age extraction - catch more patterns
            age_patterns = [
                r'I am (\d+)(?:\s+years old)?',
                r'I\'m (\d+)(?:\s+years old)?',
                r'my age is (\d+)',
                r'my age: (\d+)',
            ]

            for pattern in age_patterns:
                age_match = re.search(pattern, content, re.IGNORECASE)
                if age_match:
                    age = age_match.group(1)
                    self.logger.info(f"Detected user age: {age}")
                    self.memory.store_fact(f"user:{sender}", "age", age)
                    break

            # Basic location extraction
            location_match = re.search(r'I(?:\'m| am) (?:from|in) ([A-Za-z\s,]+)', content)
            if location_match:
                location = location_match.group(1).strip()
                self.logger.info(f"Detected user location: {location}")
                self.memory.store_fact(f"user:{sender}", "location", location)

            # Basic preference extraction
            like_match = re.search(r'I (?:like|love|enjoy) ([A-Za-z\s,]+)', content)
            if like_match:
                preference = like_match.group(1).strip()
                self.logger.info(f"Detected user preference: {preference}")
                self.memory.store_fact(f"user:{sender}", "likes", preference)

            # Basic dislike extraction
            dislike_match = re.search(r'I (?:dislike|hate|don\'t like) ([A-Za-z\s,]+)', content)
            if dislike_match:
                dislike = dislike_match.group(1).strip()
                self.logger.info(f"Detected user dislike: {dislike}")
                self.memory.store_fact(f"user:{sender}", "dislikes", dislike)

        except Exception as e:
            self.logger.error(f"Error extracting facts: {e}")


    def _summarize_and_store_conversation(self, max_chunks=10):
        """
        Summarize recent conversation and store it in long-term memory.

        Args:
            max_chunks: Maximum number of conversation chunks to summarize
        """
        try:
            # Get recent conversation chunks
            conversation_chunks = self.memory.get_recent_conversation(limit=max_chunks)

            if not conversation_chunks or len(conversation_chunks) < 3:
                # Not enough conversation to summarize
                return

            # Format conversation for the LLM
            conversation_text = ""
            for item in conversation_chunks:
                speaker = "User" if item['speaker'] != 'agent' else "Assistant"
                conversation_text += f"{speaker}: {item['content']}\n"

            # Create prompt for summarization
            prompt = f"""
            Summarize the following conversation in 2-3 sentences, focusing on:
            1. Key topics discussed
            2. Important information shared by the user
            3. Any commitments or actions promised

            Conversation:
            {conversation_text}

            Summary:
            """

            # Get summary from LLM
            self.logger.info("Generating conversation summary")
            result = self.llm_manager.submit_task_sync(
                model_name=self.consciousness_model,
                prompt=prompt,
                max_tokens=256,
                temperature=0.5,
                timeout=10.0
            )

            if result and 'text' in result:
                summary = result['text'].strip()

                # Extract user ID
                user_id = None
                for item in conversation_chunks:
                    if item['speaker'] != 'agent':
                        user_id = item['speaker']
                        break

                if not user_id:
                    user_id = "unknown"

                # Generate a timestamp for this conversation
                timestamp = time.time()
                conversation_id = f"conv_{int(timestamp)}"

                # Get emotional tone
                emotion_data = self._extract_emotional_tone(conversation_chunks)

                # Store summary in memory
                self.logger.info(f"Storing conversation summary: {summary[:100]}...")

                # Create a memory entry with emotion data
                memory_content = {
                    'user_id': user_id,
                    'conversation_id': conversation_id,
                    'summary': summary,
                    'timestamp': timestamp
                }

                # Add emotion data if available
                if emotion_data:
                    memory_content['emotion'] = emotion_data

                # Use the existing episodic memory system
                self.memory.store_episodic_memory({
                    'timestamp': timestamp,
                    'episode_type': 'conversation_summary',
                    'content': json.dumps(memory_content),
                    'importance': 0.8  # High importance for conversation summaries
                })

                # If the user_id is known, also store as a user fact
                if user_id != "unknown":
                    # Get user info
                    user_info = self.memory.get_user_info(user_id=user_id)
                    if user_info:
                        user_name = user_info.get('name', 'unknown')

                        # Store formatted summary as a conversation fact
                        fact_value = f"Talked about: {summary}"

                        # Add emotion if available
                        if emotion_data:
                            tone = emotion_data.get('overall_tone', '')
                            if tone:
                                fact_value += f" (Tone: {tone})"

                        # Store fact
                        self.memory.store_fact(
                            entity=f"user:{user_id}",
                            attribute=f"conversation_{int(timestamp)}",
                            value=fact_value
                        )

                        self.logger.info(f"Stored summary as fact for user {user_name}")

                return summary

            return None

        except Exception as e:
            self.logger.error(f"Error summarizing conversation: {e}")
            self.logger.error(traceback.format_exc())
            return None


    def _extract_emotional_tone(self, conversation_chunks):
        """
        Extract the emotional tone of a conversation.

        Args:
            conversation_chunks: List of conversation chunks

        Returns:
            Dict with emotional tone assessment
        """
        try:
            if not conversation_chunks or len(conversation_chunks) < 2:
                return None

            # Format conversation for the LLM
            conversation_text = ""
            for item in conversation_chunks:
                speaker = "User" if item['speaker'] != 'agent' else "Assistant"
                conversation_text += f"{speaker}: {item['content']}\n"

            # Create prompt for emotion analysis
            prompt = f"""
            Analyze the emotional tone of the following conversation:

            {conversation_text}

            For each participant (User and Assistant), identify:
            1. The primary emotion (e.g., happy, curious, frustrated, neutral)
            2. The intensity of that emotion (low, medium, high)
            3. Any shifts in emotion during the conversation

            Provide your analysis in JSON format with these keys:
            - "user_emotion": Primary emotion of the user
            - "user_intensity": Intensity of user's emotion
            - "assistant_emotion": Primary emotion of the assistant
            - "assistant_intensity": Intensity of assistant's emotion
            - "overall_tone": Overall tone of the conversation (e.g., positive, neutral, negative)
            """

            # Get analysis from LLM
            self.logger.info("Analyzing emotional tone of conversation")
            result = self.llm_manager.submit_task_sync(
                model_name=self.consciousness_model,
                prompt=prompt,
                max_tokens=256,
                temperature=0.4,
                timeout=10.0
            )

            if result and 'text' in result:
                # Try to parse JSON
                try:
                    # Find JSON in text
                    text = result['text']
                    start_idx = text.find('{')
                    end_idx = text.rfind('}')

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = text[start_idx:end_idx+1]
                        emotion_data = json.loads(json_str)
                        return emotion_data
                except:
                    # If JSON parsing fails, return None
                    self.logger.error("Failed to parse emotional tone as JSON")
                    return None

            return None

        except Exception as e:
            self.logger.error(f"Error extracting emotional tone: {e}")
            self.logger.error(traceback.format_exc())
            return None


    def _identify_conversation_topics(self, message_content):
        """
        Identify the topics in a message for better memory retrieval.

        Args:
            message_content: The message content to analyze

        Returns:
            List of key topics
        """
        try:
            # Create prompt for topic extraction
            prompt = f"""
            Extract 2-3 key topics from this message:

            "{message_content}"

            Provide only the topics, separated by commas, with no additional text.
            """

            # Get topics from LLM
            result = self.llm_manager.submit_task_sync(
                model_name=self.consciousness_model,
                prompt=prompt,
                max_tokens=64,
                temperature=0.3,
                timeout=5.0
            )

            if result and 'text' in result:
                # Extract topics
                topics = [topic.strip() for topic in result['text'].split(',')]
                return topics

            return []

        except Exception as e:
            self.logger.error(f"Error identifying conversation topics: {e}")
            return []
