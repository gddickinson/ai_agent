embodied_agent/
├── core/
│   ├── __init__.py
│   ├── agent.py             # Main agent orchestration
│   ├── consciousness.py     # "Conscious" interface LLM
│   ├── perception.py        # Sensory processing modules
│   ├── cognition.py         # Reasoning, emotions, planning
│   └── memory.py            # Memory management system
├── hardware/
│   ├── __init__.py
│   ├── sensors.py           # Sensor interface abstraction
│   ├── actuators.py         # Output/action interfaces
│   └── adapters/            # Hardware-specific implementations
│       ├── __init__.py
│       ├── mac_adapters.py
│       └── robot_adapters.py
├── llm/
│   ├── __init__.py
│   ├── manager.py           # LLM orchestration and threading
│   └── models.py            # Model definitions and interfaces
├── utils/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   └── logging.py           # Logging utilities
├── main.py                  # Application entry point
└── config.yaml              # Configuration parameters
