"""Local-only data for the dashboard: the agent's memory database, its output logs and the local Ollama server.

These read George's own files (memory/memory.db, memory/output/*.txt, logs/*.log) and are never exported to the
static site. The database is opened read-only.
"""
import sqlite3
import time
from pathlib import Path

import requests

from web.introspect import ROOT, load_config

TABLES = ["perceptions", "episodic_memories", "episodic_memory", "facts", "semantic_memories", "semantic_memory",
          "users", "conversations"]


def _memory_dir() -> Path:
    d = Path(load_config().get("memory", {}).get("memory_dir", "memory"))
    return d if d.is_absolute() else ROOT / d


def _connect():
    db = _memory_dir() / "memory.db"
    if not db.exists():
        return None
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def memory_overview():
    """Row counts per table (the same tables MemoryManager.get_memory_stats counts)."""
    conn = _connect()
    if conn is None:
        return {"path": str(_memory_dir() / "memory.db"), "exists": False, "counts": {}}
    try:
        counts = {}
        for t in TABLES:
            try:
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[t] = None
        return {"path": str(_memory_dir() / "memory.db"), "exists": True, "counts": counts}
    finally:
        conn.close()


def memory_rows(table: str, limit: int = 50):
    """Newest rows of one memory table (KeyError for anything not in TABLES)."""
    if table not in TABLES:
        raise KeyError(table)
    conn = _connect()
    if conn is None:
        return {"columns": [], "rows": []}
    try:
        cur = conn.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", (max(1, min(int(limit), 500)),))
        return {"columns": [c[0] for c in cur.description], "rows": [list(r) for r in cur.fetchall()]}
    except sqlite3.OperationalError:
        return {"columns": [], "rows": []}
    finally:
        conn.close()


def _tail(path: Path, lines: int):
    try:
        text = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return text[-lines:]


def output_logs(lines: int = 60):
    """Tails of the simulated display/speaker output files (config.yaml output_file) and any logs/*.log."""
    cfg = load_config()
    files = []
    for acts in cfg["hardware"]["actuators"].values():
        for a in acts or []:
            if a.get("output_file"):
                p = ROOT / a["output_file"]
                files.append({"name": f"{a['name']} ({a['type']})", "path": a["output_file"], "exists": p.exists(),
                              "lines": _tail(p, lines)})
    for p in sorted((ROOT / "logs").glob("*.log")):
        files.append({"name": p.name, "path": p.relative_to(ROOT).as_posix(), "exists": True, "lines": _tail(p, lines)})
    return {"files": files, "read_at": time.time()}


def ollama_status(models: dict, timeout: float = 1.5):
    """Ask each configured Ollama api_base (localhost by default) which models it has."""
    bases = sorted({m.get("api_base", "http://localhost:11434/api") for m in models.values()})
    servers = []
    for base in bases:
        try:
            r = requests.get(f"{base}/tags", timeout=timeout)
            r.raise_for_status()
            names = sorted(m.get("name", "") for m in r.json().get("models", []))
            servers.append({"api_base": base, "up": True, "installed": names})
        except Exception as e:  # noqa: BLE001 - any failure means "not reachable"
            servers.append({"api_base": base, "up": False, "error": type(e).__name__, "installed": []})
    installed = {n for s in servers for n in s["installed"]}
    roles = [{"role": k, "model_id": m.get("model_id"), "api_base": m.get("api_base"),
              "installed": m.get("model_id") in installed} for k, m in models.items()]
    return {"servers": servers, "roles": roles}
