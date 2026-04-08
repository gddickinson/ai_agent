#!/usr/bin/env python3
"""
Embodied AI Agent - Main Entry Point
This script initializes and runs the embodied AI agent system.
"""

import argparse
import logging
import signal
import sys
import yaml
import time
import os
from pathlib import Path

from core.agent import EmbodiedAgent
from utils.config import load_config
from utils.logging import setup_logging

# Global reference for signal handler cleanup
_agent = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Embodied AI Agent")
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
    return parser.parse_args()

def _signal_handler(signum: int, frame) -> None:
    """Handle termination signals for graceful shutdown."""
    global _agent
    sig_name = signal.Signals(signum).name
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
    if _agent is not None and _agent.is_running:
        _agent.stop()
    sys.exit(0)


def main():
    """Main entry point for the application."""
    global _agent

    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("Starting Embodied AI Agent")

    try:
        # Load configuration
        config = load_config(args.config)

        # Override hardware platform from command line if specified
        if args.hardware:
            config['hardware']['platform'] = args.hardware

        # Initialize agent
        agent = EmbodiedAgent(config)
        _agent = agent

        # Start agent
        agent.start()

        # Main loop
        try:
            while True:
                time.sleep(0.1)  # Prevent high CPU usage in main thread

                # Check if agent has stopped
                if not agent.is_running:
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected")
        finally:
            # Cleanup
            agent.stop()
            _agent = None
            logger.info("Agent stopped")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
