"""Facts about the framework read from its own code and config, for the web dashboard and the static site.

Nothing here calls an LLM or touches hardware: modules are parsed with `ast`, the memory schema is created by the
real MemoryManager in a temporary directory, and the cognitive defaults come from constructing the real modules
with stand-in LLM/memory objects.
"""
import ast
import sqlite3
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"
PACKAGES = ["core", "hardware", "llm", "utils", "gui"]
ENTRY_POINTS = ["main.py", "enhanced_chat.py"]
INTERNAL = {"core", "hardware", "llm", "utils", "gui"}


def load_config():
    """config.yaml as the agent reads it (via utils.config.load_config: same validation and defaults)."""
    from utils.config import load_config as _load
    return _load(str(CONFIG))


def _first_line(doc):
    return (doc or "").strip().splitlines()[0].strip() if doc and doc.strip() else ""


def _module_info(path: Path):
    src = path.read_text()
    tree = ast.parse(src)
    classes, functions, imports = [], [], set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append({"name": node.name, "doc": _first_line(ast.get_docstring(node)),
                            "bases": [ast.unparse(b) for b in node.bases],
                            "methods": len(methods), "public": [m for m in methods if not m.startswith("_")]})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in INTERNAL:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names if a.name.split(".")[0] in INTERNAL)
    return {"path": path.relative_to(ROOT).as_posix(), "lines": len(src.splitlines()),
            "doc": _first_line(ast.get_docstring(tree)), "classes": classes, "functions": functions,
            "imports": sorted(imports)}


def architecture():
    """Every module of the framework (packages + entry points) with its classes, functions and internal imports."""
    files = [ROOT / f for f in ENTRY_POINTS]
    for pkg in PACKAGES:
        files += sorted(p for p in (ROOT / pkg).rglob("*.py") if p.name != "__init__.py")
    return [_module_info(f) for f in files if f.exists()]


def memory_schema():
    """Tables, columns and indexes that MemoryManager creates in its SQLite database."""
    from core.memory import MemoryManager
    with tempfile.TemporaryDirectory() as tmp:
        mm = MemoryManager({"memory_dir": tmp})
        mm.stop()
        conn = sqlite3.connect(Path(tmp) / "memory.db")
        try:
            tables = []
            for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                        "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"):
                cols = [{"name": r[1], "type": r[2], "pk": bool(r[5])} for r in conn.execute(f"PRAGMA table_info({name})")]
                idx = [r[1] for r in conn.execute(f"PRAGMA index_list({name})") if not r[1].startswith("sqlite_")]
                tables.append({"name": name, "columns": cols, "indexes": idx})
            return tables
        finally:
            conn.close()


class _NoLLM:
    """Stand-in LLM manager: records nothing, answers nothing (used only to construct modules)."""
    models = {}

    def submit_task(self, *a, **kw):
        return None

    def submit_task_sync(self, *a, **kw):
        return None


class _NoMemory:
    def get_working_memory(self):
        return []

    def __getattr__(self, name):
        return lambda *a, **kw: None


def defaults():
    """Initial cognitive state and fixed vocabularies, read from the real modules."""
    from core.autonomy import DEFAULT_TOPICS, AutonomyModule
    from core.cognition import CognitionModule
    from core.consciousness import ConsciousnessModule
    from core.perception import PerceptionModule
    from hardware.sensors import GenericSensor, MicrophoneSensor, CameraSensor

    cfg = load_config()
    cog = CognitionModule(cfg.get("cognition", {}), _NoLLM(), _NoMemory())
    aut = AutonomyModule(cfg.get("autonomy", {}), _NoLLM(), _NoMemory(), None)
    con = ConsciousnessModule(cfg.get("consciousness", {}), _NoLLM(), _NoMemory(), None)
    per = PerceptionModule({}, _NoLLM(), _NoMemory())
    state = cog.get_internal_state()
    return {
        "emotions": state["emotions"], "drives": state["drives"],
        "thinking_modes": aut.thinking_modes, "initial_thinking_mode": aut.current_thinking_mode,
        "topics": DEFAULT_TOPICS, "min_interest_threshold": aut.min_interest_threshold,
        "thought_interval": aut.thought_interval, "idle_threshold": aut.idle_threshold,
        "metacognition_interval": aut.metacognition_interval,
        "consciousness": {"max_monologue_length": con.max_monologue_length,
                          "min_initiation_interval": con.min_initiation_interval,
                          "idle_initiation_threshold": con.idle_initiation_threshold,
                          "initiative_startup_delay": con.initiative_startup_delay,
                          "fallback_response": con._generate_fallback_response({"id": "m", "sender": "human"})["content"],
                          "default_state": con._get_cognition_state()},
        "perception_default_intervals": per.process_intervals,
        "sensor_default_data": {
            "camera": CameraSensor({}).simulate_data,
            "microphone": MicrophoneSensor({}).simulate_data,
            "generic": {"min": GenericSensor({}).simulate_min, "max": GenericSensor({}).simulate_max},
        },
    }


def config_summary():
    """The parts of config.yaml the dashboard shows: platforms, sensors, actuators, models, intervals."""
    cfg = load_config()
    hw = cfg["hardware"]
    platforms = sorted({k for k in hw["sensors"] if k != "default"} | {k for k in hw["actuators"] if k != "default"})
    return {
        "platform": hw["platform"], "platforms": platforms,
        "sensors": {p: hw["sensors"].get(p, []) for p in platforms},
        "actuators": {p: hw["actuators"].get(p, []) for p in platforms},
        "models": cfg["llm"]["models"], "num_workers": cfg["llm"].get("num_workers"),
        "memory": cfg.get("memory", {}), "perception": cfg.get("perception", {}),
        "cognition": cfg.get("cognition", {}), "consciousness": cfg.get("consciousness", {}),
        "autonomy": cfg.get("autonomy", {}), "processing": cfg.get("processing", {}),
    }


def raw_config_text():
    return CONFIG.read_text()


def readme_summary():
    """First paragraph of the README (the project description)."""
    for line in (ROOT / "README.md").read_text().splitlines()[1:]:
        if line.strip():
            return line.strip()
    return ""


__all__ = ["architecture", "memory_schema", "defaults", "config_summary", "load_config", "readme_summary",
           "raw_config_text", "yaml"]
