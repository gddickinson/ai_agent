"""
Hardware abstraction layer for the Embodied AI Agent.

Modules:
    sensors -- SensorManager for input devices (camera, microphone, etc.)
    actuators -- ActuatorManager for output devices (display, speakers, motors)
    adapters/ -- Platform-specific implementations (mac, robot)
"""

from hardware.sensors import SensorManager
from hardware.actuators import ActuatorManager

__all__ = ["SensorManager", "ActuatorManager"]
