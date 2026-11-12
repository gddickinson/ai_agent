# Embodied AI Agent -- Interface Map

## Entry Points
- `main.py` -- Main entry point; parses CLI args, loads config, starts EmbodiedAgent
- `enhanced_chat.py` -- Simplified chat interface with direct LLM fallbacks
- `gui/app.py` -- Tkinter GUI (`python -m gui.app`); `_archive/agent_gui.py` is the old single-file version
- `run_web.py` -- web dashboard on http://127.0.0.1:8132 (see `web/`)

## Core Modules (`core/`)
- `agent.py` -- `EmbodiedAgent` class; orchestrates all components; `_perception_cycle()` is one pass of the perception loop
- `perception.py` -- `PerceptionModule`; processes raw sensor data into meaningful perceptions using LLMs
- `cognition.py` -- `CognitionModule`; reasoning, emotion, and planning using LLMs
- `consciousness.py` -- `ConsciousnessModule`; high-level awareness, communication coordinator, message handling
- `memory.py` -- `MemoryManager`; perception memory, working memory, episodic/semantic memory (SQLite-backed)
- `autonomy.py` -- `AutonomyModule`, `DEFAULT_TOPICS`; autonomous thought generation, metacognition, topic tracking (`_thought_interval`, `_topic_weights`)

## LLM (`llm/`)
- `manager.py` -- `LLMManager` and `LLMTask`; manages multiple Ollama LLM models, task queuing, inference

## Hardware (`hardware/`)
- `sensors.py` -- `SensorManager` (`get_readings()` by name, `get_all_data()` by type); camera (OpenCV optional), microphone, generic sensors with simulation
- `actuators.py` -- `ActuatorManager`; manages outputs (display, speakers, motors)
- `motor_model.py` -- pure simulated-motor state update (`new_state`, `apply_command`, `step`, `TICK`)
- `adapters/mac_adapters.py` -- Mac-specific hardware adapters (empty placeholder)
- `adapters/robot_adapters.py` -- Robot-specific hardware adapters (empty placeholder)

## Utilities (`utils/`)
- `config.py` -- `load_config()`, `save_config()`, `merge_configs()`; YAML config handling
- `logging.py` -- `setup_logging()`, `LoggerMixin`, `log_execution_time()`; logging setup

## Web (`web/`) -- dashboard + GitHub Pages site
- `app.py` -- FastAPI app; `STATIC_ENDPOINTS` (info, config, defaults, architecture, schema) exported by `scripts/build_static.py`; `/api/local/*` local only
- `introspect.py` -- facts from the code: modules via ast, SQLite schema from MemoryManager, cognitive defaults, config summary
- `local_data.py` -- local only: read-only memory.db browser, output-log tails, Ollama status
- `agent_control.py` -- local only: run the real EmbodiedAgent (sensors forced to simulation) from the browser
- `static/` -- ES-module front end (no build). Pages: `home.js`, `world.js` (+ `world_view.js`), `architecture.js`, `mind.js`, `hardware.js`, `memory.js`, `dashboard.js`, `about.js`; `core.js`/`style.css` shared kit
- `static/*_engine.js` -- JS ports of the non-LLM logic: `actuators_engine.js` (motor model, routing), `sensors_engine.js` (simulated sensors), `mind_engine.js` (working memory, topics, thinking modes, emotions, facts, initiation); `agent_sim.js` runs them on a simulated clock
- `static/*.test.mjs` -- node tests against `tests/fixtures/engine_fixture.json`

## Scripts (`scripts/`)
- `init_script.py` -- Project directory structure initialization
- `monitor_output.py` -- Real-time monitoring of agent output files
- `make_engine_fixture.py` -- Python results the JS engines must reproduce
- `build_static.py` -- export the static site; `screenshots.py` -- browser smoke test + README images

## Tests (`tests/`)
- `test_config.py` -- Unit tests for config loading and validation
- `test_memory.py` -- Unit tests for MemoryManager
- `conftest.py` -- keeps the manual hardware/Ollama scripts out of pytest collection
- `test_motor_model.py`, `test_simulation_fallback.py`, `test_sensors_without_opencv.py`, `test_perception_pipeline.py`, `test_agent_loops.py`, `test_autonomy_logic.py`, `test_consciousness_logic.py` -- unit tests (fake LLM, no network)
- `test_web.py`, `test_engine_fixture.py` -- web API, static build, JS engine node tests, fixture is current
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
