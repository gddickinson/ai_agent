# Embodied AI Agent -- Interface Map

## Entry Points
- `main.py` -- Main entry point; parses CLI args, loads config, starts EmbodiedAgent
- `enhanced_chat.py` -- Simplified chat interface with direct LLM fallbacks
- `agent_gui.py` -- Tkinter-based GUI for interacting with the agent

## Core Modules (`core/`)
- `agent.py` -- `EmbodiedAgent` class; orchestrates all components (perception, cognition, consciousness, memory, hardware)
- `perception.py` -- `PerceptionModule`; processes raw sensor data into meaningful perceptions using LLMs
- `cognition.py` -- `CognitionModule`; reasoning, emotion, and planning using LLMs
- `consciousness.py` -- `ConsciousnessModule`; high-level awareness, communication coordinator, message handling
- `memory.py` -- `MemoryManager`; perception memory, working memory, episodic/semantic memory (SQLite-backed)
- `autonomy.py` -- `AutonomyModule`; autonomous thought generation, metacognition, topic tracking

## LLM (`llm/`)
- `manager.py` -- `LLMManager` and `LLMTask`; manages multiple Ollama LLM models, task queuing, inference

## Hardware (`hardware/`)
- `sensors.py` -- `SensorManager`; manages sensor inputs (camera, microphone, etc.)
- `actuators.py` -- `ActuatorManager`; manages outputs (display, speakers, motors)
- `adapters/mac_adapters.py` -- Mac-specific hardware adapters
- `adapters/robot_adapters.py` -- Robot-specific hardware adapters

## Utilities (`utils/`)
- `config.py` -- `load_config()`, `save_config()`, `merge_configs()`; YAML config handling
- `logging.py` -- `setup_logging()`, `LoggerMixin`, `log_execution_time()`; logging setup

## Scripts (`scripts/`)
- `init_script.py` -- Project directory structure initialization
- `monitor_output.py` -- Real-time monitoring of agent output files

## Tests (`tests/`)
- `test_config.py` -- Unit tests for config loading and validation
- `test_memory.py` -- Unit tests for MemoryManager
- `test_ollama.py` -- Ad-hoc Ollama API connectivity test
- `test_autonomy.py` -- Ad-hoc autonomy module test
- `camera_test.py` -- Ad-hoc camera test
- `vision_test.py` -- Ad-hoc vision test

## Configuration
- `config.yaml` -- Main configuration file (hardware, LLM models, memory, cognition settings)

## Data Flow
```
Sensors --> PerceptionModule --> MemoryManager --> CognitionModule --> ConsciousnessModule --> Actuators
                                     ^                                        |
                                     |                                        v
                                     +--- AutonomyModule (autonomous thoughts) ---+
```
