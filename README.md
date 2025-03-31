# Embodied AI Agent

An extensible, modular framework for creating embodied AI agents that can interact with the world through various sensors and actuators.

## Overview

This project implements a cognitive architecture for AI agents that aims to model aspects of human cognition, including:

- **Perception**: Processing sensory inputs from cameras, microphones, and other sensors
- **Memory**: Storing and retrieving experiences, knowledge, and working memory
- **Cognition**: Reasoning, planning, and emotional processing
- **Consciousness**: A high-level interface with the external world that coordinates actions

The architecture is designed to be flexible and adaptable to different hardware platforms, from desktop computers to physical robots.

## Key Features

- **Modular Design**: Each component can be developed and tested independently
- **Multi-LLM Architecture**: Uses specialized LLMs for different cognitive functions
- **Hardware Abstraction**: Adapts to available sensors and actuators on different platforms
- **Memory Systems**: Includes working memory, episodic memory, and semantic memory
- **Simulated Hardware**: Provides simulated sensors and actuators for testing and development

## Project Structure

```
embodied_agent/
├── core/                   # Core cognitive modules
│   ├── agent.py           # Main agent orchestration
│   ├── consciousness.py   # "Conscious" interface LLM
│   ├── perception.py      # Sensory processing modules
│   ├── cognition.py       # Reasoning, emotions, planning
│   └── memory.py          # Memory management system
├── hardware/               # Hardware interfaces
│   ├── sensors.py         # Sensor interface abstraction
│   ├── actuators.py       # Output/action interfaces
│   └── adapters/          # Hardware-specific implementations
│       ├── mac_adapters.py
│       └── robot_adapters.py
├── llm/                    # LLM handling
│   ├── manager.py         # LLM orchestration and threading
│   └── models.py          # Model definitions and interfaces
├── utils/                  # Utility modules
│   ├── config.py          # Configuration management
│   └── logging.py         # Logging utilities
├── memory/                 # Memory storage
│   ├── episodic/          # Episodic memory storage
│   ├── semantic/          # Semantic memory storage
│   └── output/            # Output logs
├── logs/                   # Log files
├── main.py                 # Application entry point
└── config.yaml             # Configuration parameters
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/embodied_agent.git
   cd embodied_agent
   ```

2. Set up the project structure:
   ```
   python init_project.py
   ```

3. Install dependencies:
   ```
   pip install pyyaml requests
   ```

4. Install Ollama for local LLM inference:
   - Visit [Ollama's website](https://ollama.ai/) for installation instructions
   - Pull the llama3 model: `ollama pull llama3`

## Usage

1. Run the agent:
   ```
   python main.py
   ```

2. To specify a different configuration file:
   ```
   python main.py -c custom_config.yaml
   ```

3. To run on a specific hardware platform:
   ```
   python main.py --hardware robot
   ```

4. To enable debug mode:
   ```
   python main.py --debug
   ```

## Configuration

The agent is configured through the `config.yaml` file. Some key configuration options:

- **Hardware Platform**: Configure sensors and actuators for different platforms
- **LLM Models**: Configure which models to use for different cognitive functions
- **Processing Intervals**: Set how frequently different modules should run
- **Memory Settings**: Configure memory capacities and storage

## Extending

### Adding New Sensors

1. Implement a new sensor class that extends `SensorInterface`
2. Add the sensor to the configuration file
3. Update perception handling for the new sensor type if needed

### Adding New Actuators

1. Implement a new actuator class that extends `ActuatorInterface`
2. Add the actuator to the configuration file
3. Update action handling for the new actuator type if needed

### Adapting to a New Platform

1. Create a new adapter module in `hardware/adapters/`
2. Add platform-specific sensor and actuator implementations
3. Create a configuration section for the new platform

## How It Works

The agent operates through several parallel processing loops:

1. **Perception Loop**: Polls sensors, processes the data with specialized LLMs
2. **Cognition Loop**: Reasons about perceptions, updates emotional state, plans actions
3. **Consciousness Loop**: Processes external communications, maintains internal monologue
4. **Memory Consolidation**: Periodically consolidates important memories

Data flows from perception through cognition to consciousness, with memory systems storing and providing context along the way.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by various cognitive architectures and research in artificial general intelligence
- Built using Ollama and LLM models for cognitive processing
