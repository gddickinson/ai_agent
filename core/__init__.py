"""
Core cognitive modules for the Embodied AI Agent.

Modules:
    agent -- EmbodiedAgent orchestration
    perception -- Sensory input processing
    cognition -- Reasoning, emotion, and planning
    consciousness -- High-level awareness and communication
    memory -- Memory management (working, episodic, semantic)
    autonomy -- Autonomous thought generation
"""

from core.agent import EmbodiedAgent
from core.memory import MemoryManager
from core.perception import PerceptionModule
from core.cognition import CognitionModule
from core.consciousness import ConsciousnessModule
from core.autonomy import AutonomyModule

__all__ = [
    "EmbodiedAgent",
    "MemoryManager",
    "PerceptionModule",
    "CognitionModule",
    "ConsciousnessModule",
    "AutonomyModule",
]
