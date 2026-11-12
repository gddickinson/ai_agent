# Session log

## 2026-11-12 -- web dashboard, static site, bug fixes (branch `web-interface`)

**Added**
- `web/` FastAPI dashboard (`python run_web.py`, port 8132) and a static GitHub Pages site
  (`scripts/build_static.py`, `.github/workflows/pages.yml`, CI runs `pytest tests`).
- Pages: home, simulated world, modules & data flow, cognition & autonomy, sensors & actuators, memory, dashboard
  (local only: live agent, Ollama status, memory.db browser, output logs), about.
- JS ports of the non-LLM logic (`web/static/*_engine.js`, `agent_sim.js`) held to the Python code by node tests and
  `tests/fixtures/engine_fixture.json` (`scripts/make_engine_fixture.py`; `tests/test_engine_fixture.py` checks it is
  current, rtol 1e-9).
- `hardware/motor_model.py` (pure motor model), `AutonomyModule._thought_interval/_topic_weights`,
  `ConsciousnessModule._initiation_probability`, `DEFAULT_TOPICS`, `SensorManager.get_readings()`,
  `EmbodiedAgent._perception_cycle()`.

**Bugs fixed (each in its own commit, with a regression test)**
1. pytest collected the manual webcam/Ollama scripts (2 errors, one live HTTP call) -> `tests/conftest.py`.
2. Simulated motors stopped one tick after a `move` command (snapped back to target 0).
3. Fallback to simulation for unimplemented real hardware never started a thread (running flag already set).
4. Sensor data never reached perception (`get_all_data` grouped by type into lists; perception expected by name;
   `_process_audio_data` / `_process_general_data` did not exist). Committed memory.db had 0 perceptions.
5. Autonomy: topic conversation starters were overwritten by the reflection prompt; topics named with spaces were
   boosted and decayed in the same update.
6. Conversation-initiation 5-minute startup delay was measured from 0, so never applied.
7. Simulated microphone emitted a second, hidden default phrase list instead of `.simulate_data`.
8. `EmbodiedAgent.start()` set `is_running` after starting the loop threads, so the perception, cognition and
   consciousness loops exited immediately.
- Also: OpenCV is optional for the sensor layer (camera falls back to simulation without it).

**Follow-up fixes (same day)**
9. memory/memory.db and memory/output/*.txt (personal chat logs) untracked and ignored; they stay on disk and
   **remain in earlier commits of the git history** (no history rewrite).
10. Episodic memory: one table (`episodic_memory`) for writes, reads, summaries, decay and consolidation. Schema
    version 1 (PRAGMA user_version) migrates once: adds `access_count`, merges and drops `episodic_memories`, marks
    existing episodes accessed now. Decay counts 30 days from last access, so episodes that were unreadable before
    the fix are not deleted. Checked on a copy of a real memory.db (80 episodes kept). Consolidation works.
11. LLM manager: a result with `error` (Ollama unreachable, timeout) is reported as a failure, so the agent uses its
    fallback response instead of speaking the error text.

**Still for George's decision (not changed)**
- Consciousness reads emotions from an `internal_state` working-memory item that nothing writes, so it always uses
  its built-in default state (curiosity 0.8) for prompts and initiation odds.
- Actions from the action LLM usually name a direction as target (`move` -> `forward`), which matches no motor
  (`execute_action` needs `all` or a motor name).
- Whether to purge memory/memory.db and memory/output/*.txt from the published git history.
- Several legacy files exceed the 500-line budget (consciousness.py 1.8k, memory.py 1.6k, autonomy.py, actuators.py,
  sensors.py, perception.py, cognition.py, llm/manager.py); not split here.
