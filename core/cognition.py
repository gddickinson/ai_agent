"""
Cognition Module
Handles higher-level cognitive processes like reasoning, planning, emotion, and decision-making.
"""

import logging
import time
import json
import random
from typing import Dict, Any, List, Optional

from llm.manager import LLMManager
from core.memory import MemoryManager

class CognitionModule:
    """
    Implements the cognitive processes of the agent, including reasoning,
    planning, emotional states, and decision-making.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_manager: LLMManager,
        memory_manager: MemoryManager
    ):
        """
        Initialize the cognition module.

        Args:
            config: Configuration dictionary
            llm_manager: LLM manager instance
            memory_manager: Memory manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.llm_manager = llm_manager
        self.memory = memory_manager

        # Configure cognitive LLMs
        self.reasoning_model = config.get('reasoning_model', 'reasoning_engine')
        self.emotion_model = config.get('emotion_model', 'emotion_engine')
        self.planning_model = config.get('planning_model', 'planning_engine')

        # Internal state
        self.internal_state = {
            'emotions': {
                'joy': 0.0,
                'sadness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'curiosity': 0.5,  # Start with some curiosity
                'confusion': 0.0,
            },
            'drives': {
                'exploration': 0.7,  # Start with high exploration drive
                'social': 0.5,
                'achievement': 0.3,
                'conservation': 0.2,
                'rest': 0.0
            },
            'current_goals': [],
            'current_plan': None,
            'last_updated': time.time()
        }

        # Processing flags
        self.last_reasoning_time = 0
        self.last_emotion_time = 0
        self.last_planning_time = 0

        # Processing intervals (seconds)
        self.reasoning_interval = config.get('reasoning_interval', 5.0)
        self.emotion_interval = config.get('emotion_interval', 3.0)
        self.planning_interval = config.get('planning_interval', 10.0)

        self.logger.info("Cognition module initialized")

    def process(self):
        """
        Run the cognitive processing cycle.
        Includes reasoning about perceptions, updating emotional state,
        and planning based on current goals and internal state.
        """
        current_time = time.time()

        # Check if we should update reasoning
        if current_time - self.last_reasoning_time >= self.reasoning_interval:
            self._process_reasoning()
            self.last_reasoning_time = current_time

        # Check if we should update emotional state
        if current_time - self.last_emotion_time >= self.emotion_interval:
            self._process_emotions()
            self.last_emotion_time = current_time

        # Check if we should update planning
        if current_time - self.last_planning_time >= self.planning_interval:
            self._process_planning()
            self.last_planning_time = current_time

    def _process_reasoning(self):
        """Process reasoning about recent perceptions and memories."""
        self.logger.debug("Processing reasoning")

        # Get recent perceptions
        recent_perceptions = self.memory.get_recent_perceptions(limit=5)
        if not recent_perceptions:
            return

        # Get current working memory
        working_memory = self.memory.get_working_memory()

        # Create prompt for reasoning
        prompt = self._create_reasoning_prompt(recent_perceptions, working_memory)

        # Send to LLM for processing
        def reasoning_callback(result, task):
            if result:
                try:
                    # Parse reasoning output
                    reasoning_text = result.get('text', '')
                    reasoning = self._parse_reasoning_output(reasoning_text)

                    # Update working memory with reasoning
                    if reasoning:
                        self.memory.add_to_working_memory(
                            item={
                                'type': 'reasoning',
                                'content': reasoning,
                                'timestamp': time.time()
                            },
                            importance=reasoning.get('importance', 0.5)
                        )

                        self.logger.debug(f"Added reasoning to working memory")
                except Exception as e:
                    self.logger.error(f"Error processing reasoning result: {e}")
            else:
                self.logger.error(f"Reasoning processing failed: {task.error}")

        # Submit task
        self.llm_manager.submit_task(
            model_name=self.reasoning_model,
            prompt=prompt,
            callback=reasoning_callback,
            max_tokens=1024,
            temperature=0.3
        )

    def _process_emotions(self):
        """Update emotional state based on recent events and perceptions."""
        self.logger.debug("Processing emotions")

        # Get recent perceptions and working memory
        recent_perceptions = self.memory.get_recent_perceptions(limit=3)
        working_memory = self.memory.get_working_memory()

        # Get current emotional state
        current_emotions = self.internal_state['emotions']

        # Create prompt for emotion processing
        prompt = self._create_emotion_prompt(recent_perceptions, working_memory, current_emotions)

        # Send to LLM for processing
        def emotion_callback(result, task):
            if result:
                try:
                    # Parse emotion output
                    emotion_text = result.get('text', '')
                    emotions = self._parse_emotion_output(emotion_text)

                    # Update internal emotional state
                    if emotions:
                        self.internal_state['emotions'] = emotions
                        self.internal_state['last_updated'] = time.time()

                        # Add emotion update to working memory if significant change
                        significant_change = False
                        for emotion, value in emotions.items():
                            old_value = current_emotions.get(emotion, 0.0)
                            if abs(value - old_value) > 0.3:  # Significant change threshold
                                significant_change = True
                                break

                        if significant_change:
                            self.memory.add_to_working_memory(
                                item={
                                    'type': 'emotion_change',
                                    'content': {
                                        'previous': current_emotions,
                                        'current': emotions
                                    },
                                    'timestamp': time.time()
                                },
                                importance=0.6  # Emotional changes are relatively important
                            )

                        self.logger.debug(f"Updated emotional state")
                except Exception as e:
                    self.logger.error(f"Error processing emotion result: {e}")
            else:
                self.logger.error(f"Emotion processing failed: {task.error}")

        # Submit task
        self.llm_manager.submit_task(
            model_name=self.emotion_model,
            prompt=prompt,
            callback=emotion_callback,
            max_tokens=512,
            temperature=0.4
        )

    def _process_planning(self):
        """Update current plans and goals based on internal state."""
        self.logger.debug("Processing planning")

        # Get current internal state
        current_state = self.internal_state

        # Get working memory and some episodic memories
        working_memory = self.memory.get_working_memory()
        recent_episodic = self.memory.retrieve_episodic_memories(
            min_importance=0.4,
            limit=3
        )

        # Create prompt for planning
        prompt = self._create_planning_prompt(current_state, working_memory, recent_episodic)

        # Send to LLM for processing
        def planning_callback(result, task):
            if result:
                try:
                    # Parse planning output
                    planning_text = result.get('text', '')
                    planning = self._parse_planning_output(planning_text)

                    # Update internal state with new goals and plans
                    if planning:
                        if 'goals' in planning:
                            self.internal_state['current_goals'] = planning['goals']

                        if 'plan' in planning:
                            self.internal_state['current_plan'] = planning['plan']

                        self.internal_state['last_updated'] = time.time()

                        # Add planning update to working memory
                        self.memory.add_to_working_memory(
                            item={
                                'type': 'planning',
                                'content': planning,
                                'timestamp': time.time()
                            },
                            importance=0.7  # Plans are important
                        )

                        self.logger.debug(f"Updated planning state")
                except Exception as e:
                    self.logger.error(f"Error processing planning result: {e}")
            else:
                self.logger.error(f"Planning processing failed: {task.error}")

        # Submit task
        self.llm_manager.submit_task(
            model_name=self.planning_model,
            prompt=prompt,
            callback=planning_callback,
            max_tokens=1024,
            temperature=0.3
        )

    def _create_reasoning_prompt(self, perceptions: List[Dict], working_memory: List[Dict]) -> str:
        """Create a prompt for reasoning processing."""
        # Format recent perceptions as text
        perception_text = ""
        for i, p in enumerate(perceptions):
            p_type = p.get('type', 'unknown')
            p_time = p.get('timestamp', 0)
            p_interp = p.get('interpretation', 'No interpretation available')

            perception_text += f"Perception {i+1} ({p_type} at {time.ctime(p_time)}):\n"
            perception_text += f"{p_interp}\n\n"

        # Format working memory items
        memory_text = ""
        for i, m in enumerate(working_memory):
            m_type = m.get('type', 'unknown')
            m_time = m.get('timestamp', 0)
            m_content = m.get('content', {})

            memory_text += f"Memory {i+1} ({m_type} at {time.ctime(m_time)}):\n"
            if isinstance(m_content, dict):
                memory_text += json.dumps(m_content, indent=2) + "\n\n"
            else:
                memory_text += f"{m_content}\n\n"

        # Build the prompt
        prompt = f"""
        # Reasoning Task

        You are the reasoning system of an embodied AI agent. Your task is to analyze
        recent perceptions and current working memory to draw conclusions, make inferences,
        and identify patterns or anomalies.

        ## Recent Perceptions
        {perception_text}

        ## Current Working Memory
        {memory_text}

        ## Reasoning Task
        Based on this information, please:
        1. Identify key patterns, relationships, or anomalies
        2. Draw conclusions about the agent's current situation
        3. Make inferences about things not directly observed
        4. Assign an importance score (0.0-1.0) to this reasoning

        ## Response Format
        Provide your reasoning in JSON format with these keys:
        - 'patterns': Array of identified patterns
        - 'conclusions': Array of conclusions drawn
        - 'inferences': Array of inferences made
        - 'importance': Numerical score of importance (0.0-1.0)
        """

        return prompt

    def _parse_reasoning_output(self, text: str) -> Dict[str, Any]:
        """Parse the LLM output from reasoning."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                reasoning = json.loads(json_str)
                return reasoning

            # Fallback: Just wrap the text
            return {
                'raw_output': text,
                'importance': 0.3  # Default importance
            }

        except Exception as e:
            self.logger.error(f"Error parsing reasoning output: {e}")
            return {
                'raw_output': text,
                'error': str(e),
                'importance': 0.2  # Lower importance due to parsing error
            }

    def _create_emotion_prompt(self,
                              perceptions: List[Dict],
                              working_memory: List[Dict],
                              current_emotions: Dict[str, float]) -> str:
        """Create a prompt for emotion processing."""
        # Format current emotional state
        emotion_text = ""
        for emotion, value in current_emotions.items():
            emotion_text += f"{emotion}: {value:.2f}\n"

        # Format recent perceptions briefly
        perception_text = ""
        for i, p in enumerate(perceptions):
            p_type = p.get('type', 'unknown')
            p_interp = p.get('interpretation', 'No interpretation available')

            # Keep it brief
            if isinstance(p_interp, str) and len(p_interp) > 200:
                p_interp = p_interp[:200] + "..."

            perception_text += f"Perception {i+1} ({p_type}): {p_interp}\n\n"

        # Format key working memory items briefly
        memory_text = ""
        for i, m in enumerate(working_memory[:3]):  # Just the top few items
            m_type = m.get('type', 'unknown')
            m_content = m.get('content', {})

            memory_text += f"Memory {i+1} ({m_type}): "
            if isinstance(m_content, dict):
                # Extract key fields only to keep it brief
                if 'conclusions' in m_content:
                    memory_text += f"Conclusions: {m_content['conclusions']}\n"
                elif 'raw_output' in m_content:
                    brief = m_content['raw_output']
                    if len(brief) > 100:
                        brief = brief[:100] + "..."
                    memory_text += brief + "\n"
                else:
                    memory_text += f"{str(m_content)[:100]}...\n"
            else:
                brief = str(m_content)
                if len(brief) > 100:
                    brief = brief[:100] + "..."
                memory_text += brief + "\n"

        # Build the prompt
        prompt = f"""
        # Emotion Modeling Task

        You are the emotional modeling system of an embodied AI agent. Your task is to
        update the agent's emotional state based on recent perceptions and memory.

        ## Current Emotional State
        {emotion_text}

        ## Recent Perceptions
        {perception_text}

        ## Relevant Memories
        {memory_text}

        ## Emotion Modeling Task
        Based on this information, please update the agent's emotional state.
        Consider how the recent perceptions and thoughts would impact each emotion.

        The emotions to model are:
        - joy (0.0-1.0)
        - sadness (0.0-1.0)
        - anger (0.0-1.0)
        - fear (0.0-1.0)
        - surprise (0.0-1.0)
        - curiosity (0.0-1.0)
        - confusion (0.0-1.0)

        ## Response Format
        Provide the updated emotions in JSON format with each emotion as a key
        and a value between 0.0 and 1.0.

        Example:
        {{
          "joy": 0.7,
          "sadness": 0.1,
          "anger": 0.0,
          "fear": 0.2,
          "surprise": 0.5,
          "curiosity": 0.8,
          "confusion": 0.3
        }}
        """

        return prompt

    def _parse_emotion_output(self, text: str) -> Dict[str, float]:
        """Parse the LLM output from emotion processing."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                emotions = json.loads(json_str)

                # Ensure all emotions are present and within range
                expected_emotions = [
                    'joy', 'sadness', 'anger', 'fear',
                    'surprise', 'curiosity', 'confusion'
                ]

                result = {}
                for emotion in expected_emotions:
                    value = emotions.get(emotion, 0.0)
                    # Ensure value is a float between 0 and 1
                    try:
                        value = float(value)
                        value = max(0.0, min(1.0, value))
                    except:
                        value = 0.0
                    result[emotion] = value

                return result

            # Fallback: Use existing emotions with small random changes
            result = {}
            for emotion, value in self.internal_state['emotions'].items():
                # Add small random variation
                new_value = value + (random.random() - 0.5) * 0.2
                # Clamp to range
                new_value = max(0.0, min(1.0, new_value))
                result[emotion] = new_value

            return result

        except Exception as e:
            self.logger.error(f"Error parsing emotion output: {e}")
            return self.internal_state['emotions']  # Keep existing emotions on error

    def _create_planning_prompt(self,
                               internal_state: Dict[str, Any],
                               working_memory: List[Dict],
                               episodic_memories: List[Dict]) -> str:
        """Create a prompt for planning processing."""
        # Format current emotional state
        emotion_text = ""
        for emotion, value in internal_state['emotions'].items():
            emotion_text += f"{emotion}: {value:.2f}\n"

        # Format current drives
        drive_text = ""
        for drive, value in internal_state['drives'].items():
            drive_text += f"{drive}: {value:.2f}\n"

        # Format current goals
        goal_text = ""
        for i, goal in enumerate(internal_state.get('current_goals', [])):
            goal_text += f"{i+1}. {goal}\n"
        if not goal_text:
            goal_text = "No current goals.\n"

        # Format current plan
        plan_text = str(internal_state.get('current_plan', 'No current plan.'))

        # Format key working memory items
        memory_text = ""
        for i, m in enumerate(working_memory[:5]):  # Just the top few items
            m_type = m.get('type', 'unknown')
            m_time = m.get('timestamp', 0)
            m_content = m.get('content', {})

            memory_text += f"Memory {i+1} ({m_type} at {time.ctime(m_time)}):\n"
            if isinstance(m_content, dict):
                memory_text += json.dumps(m_content, indent=2)[:300] + "...\n\n"
            else:
                memory_text += f"{str(m_content)[:300]}...\n\n"

        # Format episodic memories briefly
        episodic_text = ""
        for i, m in enumerate(episodic_memories):
            m_type = m.get('episode_type', 'unknown')
            m_time = m.get('timestamp', 0)
            m_content = m.get('content', {})

            if isinstance(m_content, str):
                try:
                    m_content = json.loads(m_content)
                except:
                    pass

            episodic_text += f"Episodic {i+1} ({m_type} at {time.ctime(m_time)}):\n"
            if isinstance(m_content, dict):
                # Try to extract key information
                if 'interpretation' in m_content:
                    brief = m_content['interpretation']
                    if len(brief) > 200:
                        brief = brief[:200] + "..."
                    episodic_text += brief + "\n\n"
                else:
                    episodic_text += json.dumps(m_content, indent=2)[:200] + "...\n\n"
            else:
                brief = str(m_content)
                if len(brief) > 200:
                    brief = brief[:200] + "..."
                episodic_text += brief + "\n\n"

        # Build the prompt
        prompt = f"""
        # Planning Task

        You are the planning system of an embodied AI agent. Your task is to
        update the agent's goals and plans based on its current internal state,
        working memory, and episodic memories.

        ## Current Internal State

        ### Emotions
        {emotion_text}

        ### Drives
        {drive_text}

        ### Current Goals
        {goal_text}

        ### Current Plan
        {plan_text}

        ## Working Memory
        {memory_text}

        ## Relevant Episodic Memories
        {episodic_text}

        ## Planning Task
        Based on this information, please:
        1. Update or create new goals for the agent
        2. Create a plan to achieve these goals
        3. Consider the agent's emotional state and drives in your planning

        ## Response Format
        Provide your planning output in JSON format with these keys:
        - 'goals': Array of goal descriptions (3-5 goals)
        - 'plan': Object with steps to achieve the goals
          - 'short_term': Array of immediate actions
          - 'medium_term': Array of next steps
          - 'long_term': Array of eventual actions
        - 'priority': The current highest priority (goal or action)
        """

        return prompt

    def _parse_planning_output(self, text: str) -> Dict[str, Any]:
        """Parse the LLM output from planning."""
        try:
            # Try to find JSON in the output
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                planning = json.loads(json_str)
                return planning

            # Fallback: Just wrap the text
            return {
                'raw_output': text,
                'goals': self.internal_state.get('current_goals', []),
                'plan': self.internal_state.get('current_plan', {})
            }

        except Exception as e:
            self.logger.error(f"Error parsing planning output: {e}")
            return {
                'raw_output': text,
                'error': str(e),
                'goals': self.internal_state.get('current_goals', []),
                'plan': self.internal_state.get('current_plan', {})
            }

    def get_internal_state(self) -> Dict[str, Any]:
        """Get the current internal state."""
        return self.internal_state

    def stop(self):
        """Stop the cognition module."""
        self.logger.info("Stopping cognition module")
        # Nothing to stop in this implementation
