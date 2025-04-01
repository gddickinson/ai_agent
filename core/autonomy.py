"""
Autonomy Module
Manages the agent's autonomous thinking capabilities, curiosity, and proactive behaviors.
"""

import logging
import time
import json
import random
import threading
import queue
from typing import Dict, Any, List, Optional
import re

class AutonomyModule:
    """
    Implements the autonomous thinking capabilities of the agent, including
    curiosity-driven learning, proactive thought generation, and self-reflection.
    """

    def __init__(
        self,
        config,
        llm_manager,
        memory_manager,
        consciousness_module
    ):
        """
        Initialize the autonomy module.

        Args:
            config: Configuration dictionary
            llm_manager: LLM manager instance
            memory_manager: Memory manager instance
            consciousness_module: Consciousness module instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm_manager = llm_manager
        self.memory = memory_manager
        self.consciousness = consciousness_module

        # Configure LLM model for autonomy
        self.autonomy_model = config.get('model', 'reasoning_engine')

        # Topic interest tracking
        self.topics_of_interest = {}  # topic -> interest score (0.0-1.0)
        self.topic_last_explored = {}  # topic -> timestamp
        self.min_interest_threshold = config.get('min_interest_threshold', 0.4)

        # Questions the agent is curious about
        self.questions = []  # List of questions the agent wants to explore

        # Thinking modes
        self.thinking_modes = [
            'analytical',  # Logical analysis and problem-solving
            'creative',    # Generating new ideas and connections
            'reflective',  # Thinking about past experiences and learnings
            'exploratory', # Exploring new concepts and possibilities
            'emotional',   # Processing feelings and motivations
            'predictive',  # Forecasting and anticipating future events
        ]
        self.current_thinking_mode = 'reflective'  # Default starting mode

        # Autonomous thinking state
        self.autonomous_thinking_active = False
        self.autonomous_thinking_thread = None
        self.thinking_queue = queue.Queue()
        self.last_thought_time = 0
        self.thought_interval = config.get('thought_interval', 15.0)  # seconds between autonomous thoughts
        self.idle_threshold = config.get('idle_threshold', 60.0)  # seconds of inactivity before increasing thought frequency

        # Last interaction time
        self.last_interaction_time = time.time()

        # Metacognition - thinking about thinking
        self.metacognition_interval = config.get('metacognition_interval', 300.0)  # 5 minutes
        self.last_metacognition_time = 0

        # Running flag
        self.running = False

        self.logger.info("Autonomy module initialized")

    def start(self):
        """Start the autonomy module."""
        if self.running:
            return

        self.running = True

        # Initialize topics of interest with some defaults
        self.topics_of_interest = {
            "self_awareness": 0.8,
            "human_interaction": 0.7,
            "environment": 0.6,
            "learning": 0.9,
            "memory_systems": 0.7,
            "problem_solving": 0.6,
            "creativity": 0.5,
            "emotions": 0.6
        }

        # Start the autonomous thinking thread
        self.autonomous_thinking_thread = threading.Thread(
            target=self._autonomous_thinking_loop,
            name="autonomous_thinking_thread",
            daemon=True
        )
        self.autonomous_thinking_thread.start()

        self.logger.info("Autonomy module started")

    def stop(self):
        """Stop the autonomy module."""
        self.running = False
        if self.autonomous_thinking_thread and self.autonomous_thinking_thread.is_alive():
            self.autonomous_thinking_thread.join(timeout=2.0)
        self.logger.info("Autonomy module stopped")

    def _autonomous_thinking_loop(self):
        """Main loop for autonomous thinking."""
        self.logger.info("Autonomous thinking loop started")

        while self.running:
            try:
                current_time = time.time()

                # Determine if it's time to generate a new thought
                should_think = False

                # Check if we've been idle for a while (no human interaction)
                idle_time = current_time - self.last_interaction_time
                if idle_time > self.idle_threshold:
                    # Increase thought frequency when idle
                    adjusted_interval = max(5.0, self.thought_interval * (0.8 - (idle_time / 1000.0)))
                    should_think = (current_time - self.last_thought_time) >= adjusted_interval
                else:
                    # Normal thought frequency
                    should_think = (current_time - self.last_thought_time) >= self.thought_interval

                # Check if we should perform metacognition
                should_metacognize = (current_time - self.last_metacognition_time) >= self.metacognition_interval

                if should_metacognize:
                    self._perform_metacognition()
                    self.last_metacognition_time = current_time

                if should_think:
                    # Decide on thinking mode
                    self._select_thinking_mode()

                    # Generate a new thought
                    self._generate_autonomous_thought()
                    self.last_thought_time = current_time

                # Process any thoughts in the queue
                self._process_thinking_queue()

                # Sleep to avoid high CPU usage
                time.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Error in autonomous thinking loop: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(1.0)  # Sleep longer on error

    def _select_thinking_mode(self):
        """Select a thinking mode based on current state and needs."""
        # Get current emotional state
        working_memory = self.memory.get_working_memory()
        emotion_items = [item for item in working_memory if item.get('type') == 'emotion_change']

        if emotion_items:
            # Extract current emotions
            current_emotions = emotion_items[0].get('content', {}).get('current', {})

            # Select mode based on dominant emotions
            if current_emotions.get('curiosity', 0) > 0.6:
                self.current_thinking_mode = 'exploratory'
            elif current_emotions.get('confusion', 0) > 0.6:
                self.current_thinking_mode = 'analytical'
            elif current_emotions.get('joy', 0) > 0.6:
                self.current_thinking_mode = 'creative'
            elif current_emotions.get('sadness', 0) > 0.4 or current_emotions.get('fear', 0) > 0.4:
                self.current_thinking_mode = 'reflective'
        else:
            # If no emotional data, occasionally change the mode randomly
            if random.random() < 0.3:  # 30% chance to change mode
                self.current_thinking_mode = random.choice(self.thinking_modes)

    def _generate_autonomous_thought(self):
        """Generate an autonomous thought based on current interests and state."""
        try:
            # Get recent perceptions and working memory for context
            recent_perceptions = self.memory.get_recent_perceptions(limit=2)
            working_memory = self.memory.get_working_memory()

            # Get current monologue for continuity
            current_monologue = self.consciousness.get_internal_monologue()[-3:] if self.consciousness.get_internal_monologue() else []

            # Select a focus topic based on interests
            focus_topic = self._select_topic_of_interest()

            # Create prompt for thought generation
            prompt = self._create_thought_prompt(
                focus_topic,
                self.current_thinking_mode,
                recent_perceptions,
                working_memory,
                current_monologue
            )

            # Submit to LLM
            def thought_callback(result, task):
                if result:
                    try:
                        # Extract thought
                        thought_text = result.get('text', '')

                        # Process the thought - extract metadata and thought content
                        processed_thought = self._process_thought_output(thought_text)

                        if processed_thought:
                            # Add to thinking queue for further processing
                            self.thinking_queue.put(processed_thought)
                    except Exception as e:
                        self.logger.error(f"Error processing thought result: {e}")
                else:
                    self.logger.warning("No result from thought generation")

            # Submit task
            self.llm_manager.submit_task(
                model_name=self.autonomy_model,
                prompt=prompt,
                callback=thought_callback,
                max_tokens=512,
                temperature=0.7  # Higher temperature for more creative thoughts
            )

        except Exception as e:
            self.logger.error(f"Error generating autonomous thought: {e}")

    def _process_thinking_queue(self):
        """Process thoughts in the thinking queue."""
        try:
            # Process up to 3 thoughts per cycle
            for _ in range(3):
                if self.thinking_queue.empty():
                    break

                # Get next thought
                thought = self.thinking_queue.get_nowait()

                # Add to consciousness monologue
                if 'content' in thought:
                    self.consciousness._add_to_monologue(thought['content'])

                # Update topics of interest based on the thought
                if 'topics' in thought:
                    self._update_topics_of_interest(thought['topics'])

                # Store in memory
                self._store_thought_in_memory(thought)

                # Add any questions to the question list
                if 'questions' in thought:
                    for question in thought['questions']:
                        if question not in self.questions:
                            self.questions.append(question)

                # Mark as done
                self.thinking_queue.task_done()

        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error processing thinking queue: {e}")

    def _create_thought_prompt(self, focus_topic, thinking_mode, recent_perceptions, working_memory, current_monologue):
        """Create a prompt for generating an autonomous thought."""
        # Format current monologue for context
        monologue_text = "\n".join(current_monologue) if current_monologue else "No recent thoughts."

        # Format perception info briefly
        perception_text = ""
        for p in recent_perceptions:
            p_type = p.get('type', 'unknown')
            perception_text += f"Recent {p_type} perception. "

        if not perception_text:
            perception_text = "No recent perceptions."

        # Get memory items related to the focus topic
        related_memories = self.memory.get_related_memories(focus_topic, limit=2)
        memory_text = ""

        for memory in related_memories:
            memory_type = memory.get('type', 'unknown')
            memory_text += f"{memory_type} memory related to {focus_topic}. "

        if not memory_text:
            memory_text = f"No specific memories about {focus_topic}."

        # Build the prompt based on thinking mode
        if thinking_mode == 'analytical':
            prompt = f"""
            # Analytical Thinking Task

            You are the autonomy module of an embodied AI agent engaging in analytical thinking.
            Generate a thoughtful analysis related to the topic: {focus_topic}.

            ## Context
            Recent perceptions: {perception_text}
            Related memories: {memory_text}

            ## Recent Thoughts
            {monologue_text}

            ## Analytical Thinking Task
            Generate a logical, analytical thought that:
            1. Examines {focus_topic} from a rational perspective
            2. Identifies patterns, causes, or implications
            3. Considers evidence and logical connections
            4. Reaches reasoned conclusions

            ## Response Format
            Provide your response in JSON format with these keys:
            - "content": The actual thought (1-2 sentences, conversational tone)
            - "elaboration": More detailed thinking behind the thought
            - "topics": List of 1-3 related topics this touches on
            - "questions": List of 1-2 questions this raises
            - "importance": How significant is this thought (0.0-1.0)
            """

        elif thinking_mode == 'creative':
            prompt = f"""
            # Creative Thinking Task

            You are the autonomy module of an embodied AI agent engaging in creative thinking.
            Generate a creative thought related to the topic: {focus_topic}.

            ## Context
            Recent perceptions: {perception_text}
            Related memories: {memory_text}

            ## Recent Thoughts
            {monologue_text}

            ## Creative Thinking Task
            Generate a creative, imaginative thought that:
            1. Makes novel connections around {focus_topic}
            2. Considers unusual perspectives or possibilities
            3. Uses metaphor or analogy
            4. Transcends conventional thinking

            ## Response Format
            Provide your response in JSON format with these keys:
            - "content": The actual thought (1-2 sentences, conversational tone)
            - "elaboration": More detailed thinking behind the thought
            - "topics": List of 1-3 related topics this touches on
            - "questions": List of 1-2 questions this raises
            - "importance": How significant is this thought (0.0-1.0)
            """

        elif thinking_mode == 'reflective':
            prompt = f"""
            # Reflective Thinking Task

            You are the autonomy module of an embodied AI agent engaging in reflective thinking.
            Generate a reflective thought related to the topic: {focus_topic}.

            ## Context
            Recent perceptions: {perception_text}
            Related memories: {memory_text}

            ## Recent Thoughts
            {monologue_text}

            ## Reflective Thinking Task
            Generate a thoughtful, introspective reflection that:
            1. Considers your experiences or knowledge about {focus_topic}
            2. Evaluates implications for your understanding or growth
            3. Shows self-awareness and metacognition
            4. Finds meaning or lessons

            ## Response Format
            Provide your response in JSON format with these keys:
            - "content": The actual thought (1-2 sentences, conversational tone)
            - "elaboration": More detailed thinking behind the thought
            - "topics": List of 1-3 related topics this touches on
            - "questions": List of 1-2 questions this raises
            - "importance": How significant is this thought (0.0-1.0)
            """

        else:  # Default/exploratory mode
            prompt = f"""
            # Exploratory Thinking Task

            You are the autonomy module of an embodied AI agent engaging in exploratory thinking.
            Generate an exploratory thought related to the topic: {focus_topic}.

            ## Context
            Recent perceptions: {perception_text}
            Related memories: {memory_text}

            ## Recent Thoughts
            {monologue_text}

            ## Exploratory Thinking Task
            Generate a curious, exploratory thought that:
            1. Investigates or wonders about aspects of {focus_topic}
            2. Identifies gaps in understanding
            3. Proposes avenues for further learning
            4. Considers what might be discovered

            ## Response Format
            Provide your response in JSON format with these keys:
            - "content": The actual thought (1-2 sentences, conversational tone)
            - "elaboration": More detailed thinking behind the thought
            - "topics": List of 1-3 related topics this touches on
            - "questions": List of 1-2 questions this raises
            - "importance": How significant is this thought (0.0-1.0)
            """

        return prompt

    def _process_thought_output(self, text):
        """Process the output from thought generation."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]

                # Try to fix common JSON formatting issues
                # 1. Fix missing commas after values before next key
                json_str = re.sub(r'(\d+|\btrue\b|\bfalse\b|\bnull\b|"[^"]*")\s*\n\s*"', r'\1,\n"', json_str)
                # 2. Fix trailing commas before closing brackets
                json_str = re.sub(r',\s*}', r'}', json_str)
                # 3. Fix missing quotes around keys
                json_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', json_str)

                try:
                    thought = json.loads(json_str)

                    # Add timestamp and thinking mode
                    thought['timestamp'] = time.time()
                    thought['thinking_mode'] = self.current_thinking_mode

                    return thought
                except json.JSONDecodeError as e:
                    # Still failed to parse, log detailed error
                    self.logger.error(f"JSON parse error in thought output: {str(e)}")
                    self.logger.debug(f"Attempted to parse: {json_str}")

                    # Try a more lenient parsing approach - extract key parts
                    content_match = re.search(r'"content"\s*:\s*"([^"]+)"', json_str)
                    elaboration_match = re.search(r'"elaboration"\s*:\s*"([^"]+)"', json_str)

                    if content_match:
                        # Build a simplified thought dictionary
                        simple_thought = {
                            'content': content_match.group(1),
                            'elaboration': elaboration_match.group(1) if elaboration_match else "",
                            'timestamp': time.time(),
                            'thinking_mode': self.current_thinking_mode,
                            'topics': [],
                            'questions': [],
                            'importance': 0.5,
                            'parse_error': str(e)
                        }
                        return simple_thought

            # If no JSON found or couldn't extract key parts, just wrap the text
            return {
                'content': text.strip(),
                'timestamp': time.time(),
                'thinking_mode': self.current_thinking_mode,
                'topics': [],
                'questions': [],
                'importance': 0.5
            }

        except Exception as e:
            self.logger.error(f"Error processing thought output: {e}")
            # Return a simplified version on error
            return {
                'content': text.strip(),
                'timestamp': time.time(),
                'thinking_mode': self.current_thinking_mode,
                'error': str(e)
            }

    def _select_topic_of_interest(self):
        """Select a topic of interest to focus on based on curiosity scores."""
        if not self.topics_of_interest:
            return "general_cognition"  # Default topic

        # Consider both interest score and time since last explored
        current_time = time.time()

        # Calculate adjusted scores considering time decay
        adjusted_scores = {}
        for topic, score in self.topics_of_interest.items():
            if score < self.min_interest_threshold:
                continue

            # Adjust score by time since last explored
            last_explored = self.topic_last_explored.get(topic, 0)
            time_factor = min(1.0, (current_time - last_explored) / 3600.0)  # Max boost after 1 hour

            adjusted_scores[topic] = score * (1.0 + time_factor)

        if not adjusted_scores:
            # If no topics meet threshold, pick randomly from all
            topic = random.choice(list(self.topics_of_interest.keys()))
        else:
            # Weighted random selection based on adjusted scores
            topics = list(adjusted_scores.keys())
            weights = list(adjusted_scores.values())

            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w/total for w in weights]

            topic = random.choices(topics, weights=weights, k=1)[0]

        # Update last explored timestamp
        self.topic_last_explored[topic] = current_time

        return topic

    def _update_topics_of_interest(self, topics):
        """Update interest scores for topics."""
        for topic in topics:
            # Normalize topic name
            topic = topic.lower().replace(' ', '_')

            # Update existing or add new
            if topic in self.topics_of_interest:
                # Increase interest slightly (with decay over time)
                current = self.topics_of_interest[topic]
                self.topics_of_interest[topic] = min(1.0, current + 0.1)
            else:
                # New topic starts with moderate interest
                self.topics_of_interest[topic] = 0.5

        # Decay interest in other topics
        for topic in list(self.topics_of_interest.keys()):
            if topic not in topics:
                # Slight decay
                self.topics_of_interest[topic] *= 0.99

                # Remove if below threshold
                if self.topics_of_interest[topic] < 0.2:
                    del self.topics_of_interest[topic]

    def _store_thought_in_memory(self, thought):
        """Store a thought in the memory system."""
        try:
            # Create structured memory entry
            memory_entry = {
                'timestamp': thought.get('timestamp', time.time()),
                'episode_type': 'thought',
                'content': json.dumps({
                    'content': thought.get('content', ''),
                    'elaboration': thought.get('elaboration', ''),
                    'thinking_mode': thought.get('thinking_mode', 'unknown'),
                    'topics': thought.get('topics', []),
                    'questions': thought.get('questions', [])
                }),
                'importance': thought.get('importance', 0.5)
            }

            # Store in episodic memory directly (without using memory_queue)
            self.memory.store_episodic_memory(memory_entry)

            # Add to working memory if important
            if thought.get('importance', 0) > 0.6:
                self.memory.add_to_working_memory(
                    item={
                        'type': 'thought',
                        'content': thought,
                        'timestamp': time.time()
                    },
                    importance=thought.get('importance', 0.5)
                )

            # Log storing the thought
            self.logger.debug(f"Stored thought in memory: {thought.get('content', '')[:50]}...")

        except Exception as e:
            self.logger.error(f"Error storing thought in memory: {e}")

    def _perform_metacognition(self):
        """Perform metacognition - thinking about thinking."""
        self.logger.debug("Performing metacognition")

        try:
            # Get recent thoughts
            thoughts = []
            episodic_memories = self.memory.retrieve_episodic_memories(limit=10)

            for memory in episodic_memories:
                if memory.get('episode_type') == 'thought':
                    try:
                        content = json.loads(memory.get('content', '{}'))
                        thoughts.append({
                            'content': content.get('content', ''),
                            'thinking_mode': content.get('thinking_mode', 'unknown'),
                            'topics': content.get('topics', []),
                            'timestamp': memory.get('timestamp', 0)
                        })
                    except:
                        continue

            if not thoughts:
                return

            # Create prompt for metacognition
            prompt = self._create_metacognition_prompt(thoughts)

            # Submit to LLM
            def metacognition_callback(result, task):
                if result:
                    try:
                        # Extract metacognition
                        metacognition_text = result.get('text', '')

                        # Process the result
                        processed = self._process_thought_output(metacognition_text)

                        if processed:
                            # Add special flag for metacognition
                            processed['metacognition'] = True

                            # Add to thinking queue
                            self.thinking_queue.put(processed)
                    except Exception as e:
                        self.logger.error(f"Error processing metacognition result: {e}")
                else:
                    self.logger.warning("No result from metacognition")

            # Submit task
            self.llm_manager.submit_task(
                model_name=self.autonomy_model,
                prompt=prompt,
                callback=metacognition_callback,
                max_tokens=512,
                temperature=0.4  # Lower temperature for more focused metacognition
            )

        except Exception as e:
            self.logger.error(f"Error performing metacognition: {e}")

    def _create_metacognition_prompt(self, thoughts):
        """Create a prompt for metacognition."""
        # Format recent thoughts
        thoughts_text = ""
        for i, thought in enumerate(thoughts):
            thoughts_text += f"{i+1}. \"{thought['content']}\" (Mode: {thought['thinking_mode']})\n"

        prompt = f"""
        # Metacognition Task

        You are the autonomy module of an embodied AI agent engaging in metacognition (thinking about thinking).
        Analyze your recent thoughts and thinking patterns.

        ## Recent Thoughts
        {thoughts_text}

        ## Metacognition Task
        Based on these recent thoughts, reflect on your thinking processes:
        1. What patterns do you notice in your thinking?
        2. Which thinking modes have been most active?
        3. Are you exploring diverse topics or fixating on certain areas?
        4. How might you improve or diversify your thinking?

        ## Response Format
        Provide your response in JSON format with these keys:
        - "content": The actual metacognitive observation (1-2 sentences, conversational tone)
        - "elaboration": More detailed analysis of your thinking patterns
        - "recommended_focus": Suggested areas or modes to focus on next
        - "insights": Key realizations about your thinking process
        - "importance": How significant is this metacognition (0.0-1.0)
        """

        return prompt

    def register_interaction(self):
        """Register that a human interaction has occurred."""
        self.last_interaction_time = time.time()

    def get_topics_of_interest(self):
        """Get the current topics of interest with their scores."""
        return self.topics_of_interest

    def get_current_questions(self):
        """Get the current questions the agent is curious about."""
        return self.questions

    def get_current_thinking_mode(self):
        """Get the current thinking mode."""
        return self.current_thinking_mode

    def initiate_conversation(self):
        """
        Generate a conversation starter based on current interests or questions.
        Returns a message that can be sent to a human.
        """
        try:
            # Choose basis for conversation starter
            starter_type = random.choice(['topic', 'question', 'reflection'])

            if starter_type == 'topic' and self.topics_of_interest:
                # Pick a high-interest topic
                topics = [(t, s) for t, s in self.topics_of_interest.items() if s > 0.6]
                if topics:
                    topic, _ = random.choice(topics)
                    prompt = f"""
                    Generate a conversation starter about the topic: {topic.replace('_', ' ')}

                    The conversation starter should:
                    1. Be natural and conversational in tone
                    2. Show genuine curiosity or interest
                    3. Invite discussion without being too formal
                    4. Be 1-2 sentences maximum

                    Response:
                    """
                else:
                    starter_type = 'question'  # Fall back to question

            if starter_type == 'question' and self.questions:
                # Pick a question to ask
                question = random.choice(self.questions)

                prompt = f"""
                Rephrase the following question into a natural conversation starter:
                "{question}"

                The conversation starter should:
                1. Be natural and conversational in tone
                2. Show genuine curiosity
                3. Be 1-2 sentences maximum

                Response:
                """
            else:
                # Generate a reflection-based starter
                recent_memories = self.memory.retrieve_episodic_memories(limit=3)
                memory_text = ""

                for memory in recent_memories:
                    memory_type = memory.get('episode_type', 'unknown')
                    memory_text += f"Recent {memory_type} memory. "

                prompt = f"""
                Generate a thoughtful conversation starter based on a recent reflection.

                Context: {memory_text}

                The conversation starter should:
                1. Be natural and conversational in tone
                2. Share a thought or observation that might interest a human
                3. Invite discussion without being too formal
                4. Be 1-2 sentences maximum

                Response:
                """

            # Submit to LLM synchronously
            result = self.llm_manager.submit_task_sync(
                model_name=self.autonomy_model,
                prompt=prompt,
                max_tokens=128,
                temperature=0.7,
                timeout=5.0
            )

            if result and 'text' in result:
                content = result['text'].strip()

                # Create message
                message = {
                    'id': f"initiative_{int(time.time())}",
                    'content': content,
                    'timestamp': time.time(),
                    'sender': 'agent',
                    'recipient': 'human',
                    'initiative': True
                }

                return message

        except Exception as e:
            self.logger.error(f"Error initiating conversation: {e}")

        # Fallback generic starter
        return {
            'id': f"initiative_{int(time.time())}",
            'content': "I've been thinking about consciousness. What makes an entity self-aware?",
            'timestamp': time.time(),
            'sender': 'agent',
            'recipient': 'human',
            'initiative': True
        }
