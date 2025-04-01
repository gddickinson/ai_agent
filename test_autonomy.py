#!/usr/bin/env python3
"""
Test Autonomy Script
Used to test and demonstrate the agent's autonomous thinking capabilities.
"""

import time
import logging
import sys
import argparse
import json
from pathlib import Path

from core.agent import EmbodiedAgent
from utils.config import load_config
from utils.logging import setup_logging

def main():
    """Main entry point for testing autonomy."""
    parser = argparse.ArgumentParser(description="Test the agent's autonomous thinking")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=120,
        help="Time to run the test in seconds"
    )
    parser.add_argument(
        "--seed-topics",
        action="store_true",
        help="Seed the agent with initial topics to think about"
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting Autonomy Test")

    try:
        # Load configuration
        config = load_config(args.config)

        # Make sure autonomy is enabled
        if 'autonomy' not in config:
            logger.error("Autonomy module not configured in config.yaml")
            return 1

        # Adjust for faster thought generation during test
        config['autonomy']['thought_interval'] = 10.0  # Think every 10 seconds
        config['autonomy']['metacognition_interval'] = 60.0  # Metacognition every minute

        # Initialize agent
        agent = EmbodiedAgent(config)

        # Start agent
        agent.start()
        logger.info("Agent started")

        if args.seed_topics and hasattr(agent, 'autonomy') and agent.autonomy:
            # Seed with some interesting topics
            seed_topics = {
                "consciousness": 0.9,
                "self_awareness": 0.8,
                "artificial_intelligence": 0.8,
                "learning": 0.7,
                "creativity": 0.7,
                "human_cognition": 0.6,
                "perception": 0.6,
                "memory_formation": 0.7,
                "emotional_states": 0.6,
                "autonomous_systems": 0.7
            }
            
            # Update the agent's topics of interest
            agent.autonomy.topics_of_interest.update(seed_topics)
            logger.info(f"Seeded agent with {len(seed_topics)} initial topics")

        # Set up monologue tracking
        last_monologue_length = 0
        monologue_check_interval = 5.0  # Check every 5 seconds

        # Run for specified time
        logger.info(f"Running for {args.time} seconds...")
        start_time = time.time()
        next_check_time = start_time + monologue_check_interval

        try:
            while time.time() - start_time < args.time:
                # Check for new monologue entries
                current_time = time.time()
                
                if current_time >= next_check_time:
                    current_monologue = agent.consciousness.get_internal_monologue()
                    current_length = len(current_monologue)
                    
                    if current_length > last_monologue_length:
                        # Print new entries
                        for i in range(last_monologue_length, current_length):
                            print(f"\n{current_monologue[i]}")
                        
                        last_monologue_length = current_length
                    
                    next_check_time = current_time + monologue_check_interval
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")

        # Display final stats
        if hasattr(agent, 'autonomy') and agent.autonomy:
            print("\n=== Final Autonomy State ===")
            print(f"Total thoughts generated: {last_monologue_length - 1}")  # Exclude initial monologue
            
            topics = agent.autonomy.get_topics_of_interest()
            if topics:
                print("\nFinal Topics of Interest:")
                sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
                for topic, score in sorted_topics[:10]:  # Show top 10
                    print(f"  {topic.replace('_', ' ')}: {score:.2f}")
            
            questions = agent.autonomy.get_current_questions()
            if questions:
                print("\nFinal Questions:")
                for i, question in enumerate(questions[:5]):
                    print(f"  {i+1}. {question}")
        
        # Cleanup
        logger.info("Stopping agent")
        agent.stop()
        logger.info("Agent stopped")

    except Exception as e:
        logger.exception(f"Error in test: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
