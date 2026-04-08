# Embodied AI Agent

@INTERFACE.md

## Quick Start
```bash
pip install -r requirements.txt
python main.py --hardware mac
```

## Testing
```bash
pytest tests/ -v
```

## Key Dependencies
- pyyaml (config loading)
- requests (Ollama API communication)
- Ollama must be running locally for LLM features
