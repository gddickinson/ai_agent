#!/usr/bin/env python3
"""
Simple Chat Interface for Embodied AI Agent
Allows direct text interaction with the agent through the console.
"""

import os
import sys
import time
import argparse
import logging
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import EmbodiedAgent
from utils.config import load_config
from utils.logging import setup_logging

def chat_interface(agent):
    """Interactive chat interface to communicate with the agent."""
    print("\n=== Embodied AI Agent Chat Interface ===")
    print("Type your messages and press Enter to send")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'internal' to see the agent's internal monologue")
    print("Type 'status' to see the agent's current internal state")
    print("==========================================\n")
    
    try:
        while True:
            # Get user input
            user_input = input("\n> ")
            
            if user_input.lower() in ('exit', 'quit'):
                break
                
            # Special commands
            if user_input.lower() == 'internal':
                monologue = agent.consciousness.get_internal_monologue()
                print("\n=== Internal Monologue ===")
                for thought in monologue[-10:]:  # Show last 10 thoughts
                    print(thought)
                print("===========================")
                continue
                
            if user_input.lower() == 'status':
                internal_state = agent.cognition.get_internal_state()
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
                print("=====================")
                continue
            
            # Send regular message to agent
            message = {
                'id': f"msg_{int(time.time())}",
                'sender': 'human',
                'content': user_input,
                'timestamp': time.time()
            }
            
            # Send to agent's consciousness
            print(f"\nSending message to agent...")
            agent.consciousness.receive_message(message)
            
            # Wait for response with timeout
            response = None
            timeout = 20.0  # Maximum wait time
            start_time = time.time()
            
            print("Waiting for response", end="")
            sys.stdout.flush()
            
            while time.time() - start_time < timeout:
                # Check for response
                response = agent.consciousness.get_next_outgoing_message(timeout=0.5)
                if response:
                    break
                    
                # Show waiting animation
                print(".", end="")
                sys.stdout.flush()
                time.sleep(0.5)
            
            print()  # New line after waiting dots
                
            if response:
                print(f"\nAgent: {response.get('content', '')}")
                # If the agent had an internal thought about this, show it
                thought = response.get('thought')
                if thought:
                    print(f"\n(Internal thought: {thought})")
            else:
                print("\nAgent did not respond within the timeout period.")
                print("You might want to check the logs for any issues.")
    
    except KeyboardInterrupt:
        print("\nChat session terminated by user.")
    except Exception as e:
        print(f"\nError in chat interface: {e}")
    
    print("\nEnding chat session.")

def main():
    """Main entry point for the chat interface."""
    parser = argparse.ArgumentParser(description="Chat Interface for Embodied AI Agent")
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
    
    logger.info("Starting Embodied AI Agent Chat Interface")
    
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
        
        # Run chat interface
        chat_interface(agent)
        
        # Cleanup
        logger.info("Stopping agent")
        agent.stop()
        logger.info("Agent stopped")
            
    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
