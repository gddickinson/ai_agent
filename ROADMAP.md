# Embodied AI Agent -- Roadmap

## Current State
A modular cognitive architecture for embodied AI agents using Ollama LLMs. Core modules (perception, memory, cognition, consciousness, autonomy) are implemented with hardware abstraction for Mac and robot platforms. Has a GUI (`gui/app.py`), enhanced chat interface, and several test scripts. Has `requirements.txt`, `.gitignore`, proper pytest tests, and `INTERFACE.md`.

## Short-term Improvements
- [x] Add `requirements.txt` -- dependencies (pyyaml, requests, etc.) are only mentioned in README
- [x] Convert ad-hoc test scripts (`tests/test_ollama.py`, `tests/camera_test.py`) into proper pytest-based unit tests
- [ ] Add type hints throughout `core/agent.py`, `core/consciousness.py`, `core/memory.py`
- [ ] Add input validation and error handling in `core/perception.py` for missing/broken sensors
- [x] Remove hardcoded paths and snapshot JPGs from the repo root
- [ ] Add docstrings to all public methods in `core/` and `llm/` modules
- [x] Create `INTERFACE.md` mapping all modules, classes, and their connections
- [x] Add graceful shutdown handling in `main.py` (signal handlers, cleanup)

## Feature Enhancements
- [ ] Add support for additional LLM providers beyond Ollama (OpenAI, Anthropic) in `llm/manager.py`
- [ ] Implement persistent memory storage (SQLite or similar) instead of in-memory only
- [ ] Add conversation history export/import in `enhanced_chat.py`
- [ ] Build a web-based dashboard alternative to the Tkinter `agent_gui.py`
- [ ] Add emotion visualization and memory timeline in the GUI
- [ ] Implement rate limiting and token counting in `llm/manager.py`
- [ ] Add audio input/output support in `hardware/sensors.py` and `hardware/actuators.py`

## Long-term Vision
- [ ] Package as installable Python package with `setup.py`/`pyproject.toml` and entry points
- [ ] Add plugin system for custom cognitive modules (e.g., custom reasoning engines)
- [ ] Implement multi-agent support -- multiple agents communicating in shared environments
- [ ] Add benchmarking suite to measure response quality, latency, and memory efficiency
- [ ] Create Docker container for reproducible deployment
- [ ] Support cloud deployment with REST API for remote agent interaction

## Technical Debt
- [x] `enhanced_autodisc_working.py`-style files suggest incomplete refactoring -- consolidate
- [x] `monitor_output.py` and `init_script.py` at root level should be moved into `utils/` or `scripts/`
- [x] Several `__init__.py` files are likely empty -- add proper exports and docstrings
- [ ] The `memory/` directory mixes data storage with code concerns -- separate data dirs from source
- [x] `agent_gui.py` at root should move into `gui/` package for consistency
- [ ] No CI/CD pipeline -- add GitHub Actions for linting and testing
