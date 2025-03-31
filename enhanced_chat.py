#!/usr/bin/env python3
"""
Enhanced Chat Interface for Embodied AI Agent
Provides a simplified chat experience with direct LLM fallbacks.
"""

import sys
import os
import time
import argparse
import logging
import json
import requests
import threading
import queue
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import EmbodiedAgent
from utils.config import load_config
from utils.logging import setup_logging

class EnhancedChat:
    """Enhanced chat interface with direct LLM fallbacks."""
    
    def __init__(self, agent):
        """Initialize with an agent instance."""
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.response_queue = queue.Queue()
        self.api_base = "http://localhost:11434/api"
        self.model_id = "llama3:latest"
        
        # Register message callbacks
        self.agent.consciousness.register_message_callback('sent', self._message_sent_callback)
    
    def _message_sent_callback(self, message):
        """Callback for when a message is sent by the agent."""
        self.response_queue.put(message)
    
    def direct_llm_response(self, message_text):
        """Get a direct response from the LLM."""
        try:
            # Create a simple prompt
            prompt = f"""You are an AI assistant. Respond to the following message from a human:

Message: "{message_text}"

Your response should be helpful, friendly, and concise."""
            
            # Send to Ollama
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 256
                }
            }
            
            response = requests.post(f"{self.api_base}/generate", json=payload, timeout=10.0)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', "I'm having trouble generating a response right now.")
            
        except Exception as e:
            self.logger.error(f"Error in direct LLM response: {e}")
            return "I'm experiencing technical difficulties right now. Please try again later."
    
    def run(self):
        """Run the enhanced chat interface."""
        print("\n=== Enhanced Embodied AI Agent Chat Interface ===")
        print("This interface includes direct LLM fallbacks if the agent doesn't respond")
        print("Type your messages and press Enter to send")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'debug' to see debugging options")
        print("=================================================\n")
        
        try:
            while True:
                # Get user input
                user_input = input("\n> ")
                
                if user_input.lower() in ('exit', 'quit'):
                    break
                
                if user_input.lower() == 'debug':
                    self._show_debug_menu()
                    continue
                
                # Send message to agent
                print(f"\nSending message to agent...")
                
                message = {
                    'id': f"msg_{int(time.time())}",
                    'sender': 'human',
                    'content': user_input,
                    'timestamp': time.time()
                }
                
                # Send to agent's consciousness
                self.agent.consciousness.receive_message(message)
                
                # Wait for response with timeout
                response = None
                timeout = 10.0  # Maximum wait time
                start_time = time.time()
                
                print("Waiting for response", end="")
                sys.stdout.flush()
                
                while time.time() - start_time < timeout:
                    # Check for response
                    try:
                        response = self.response_queue.get(timeout=0.5)
                        break
                    except queue.Empty:
                        # Show waiting animation
                        print(".", end="")
                        sys.stdout.flush()
                
                print()  # New line after waiting dots
                
                if response:
                    print(f"\nAgent: {response.get('content', '')}")
                    
                    # If the agent had an internal thought about this, show it
                    thought = response.get('thought')
                    if thought:
                        print(f"\n(Internal thought: {thought})")
                else:
                    print("\nAgent did not respond in time. Falling back to direct LLM...")
                    direct_response = self.direct_llm_response(user_input)
                    print(f"\nDirect LLM: {direct_response}")
        
        except KeyboardInterrupt:
            print("\nChat session terminated by user.")
        except Exception as e:
            print(f"\nError in chat interface: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nEnding chat session.")
    
    def _show_debug_menu(self):
        """Show debugging options."""
        print("\n=== Debug Menu ===")
        print("1. Show internal monologue")
        print("2. Show agent state")
        print("3. Check memory")
        print("4. Test direct LLM")
        print("0. Return to chat")
        
        choice = input("\nEnter option: ")
        
        if choice == '1':
            monologue = self.agent.consciousness.get_internal_monologue()
            print("\n=== Internal Monologue ===")
            for thought in monologue:
                print(thought)
            
        elif choice == '2':
            try:
                internal_state = self.agent.cognition.get_internal_state()
                print("\n=== Internal State ===")
                print("Emotions:")
                for emotion, value in internal_state.get('emotions', {}).items():
                    print(f"  {emotion}: {value:.2f}")
                print("\nDrives:")
                for drive, value in internal_state.get('drives', {}).items():
                    print(f"  {drive}: {value:.2f}")
                print("\nGoals:")
                for goal in internal_state.get('current_goals', []):
                    print(f"  - {goal}")
            except Exception as e:
                print(f"Error retrieving state: {e}")
            
        elif choice == '3':
            try:
                working_memory = self.agent.memory.get_working_memory()
                print(f"\nWorking memory items: {len(working_memory)}")
                
                recent_perceptions = self.agent.memory.get_recent_perceptions()
                print(f"Recent perceptions: {len(recent_perceptions)}")
                
                # Show most recent working memory item
                if working_memory:
                    print("\nMost recent working memory item:")
                    item = working_memory[0]
                    print(f"Type: {item.get('type')}")
                    print(f"Timestamp: {time.ctime(item.get('timestamp', 0))}")
                    content = item.get('content', {})
                    if isinstance(content, dict):
                        print(f"Content: {json.dumps(content, indent=2)[:200]}...")
                    else:
                        print(f"Content: {str(content)[:200]}...")
            except Exception as e:
                print(f"Error checking memory: {e}")
            
        elif choice == '4':
            test_message = input("\nEnter a test message: ")
            print("Testing direct LLM response...")
            response = self.direct_llm_response(test_message)
            print(f"\nResponse: {response}")

def main():
    """Main entry point for the enhanced chat interface."""
    parser = argparse.ArgumentParser(description="Enhanced Chat Interface for Embodied AI Agent")
    parser.add_argument(
        "-c", "--config", 
        default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--hardware", 
        choices=["mac", "robot"], 
        default="mac",
        help="Hardware platform to run on"
    )
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Enhanced Embodied AI Agent Chat Interface")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override hardware platform from command line if specified
        if args.hardware:
            config['hardware']['platform'] = args.hardware
            
        # Initialize agent
        agent = EmbodiedAgent(config)
        
        # Start agent
        agent.start()
        logger.info("Agent started")
        
        # Create and run enhanced chat
        chat = EnhancedChat(agent)
        chat.run()
        
        # Cleanup
        logger.info("Stopping agent")
        agent.stop()
        logger.info("Agent stopped")
            
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
