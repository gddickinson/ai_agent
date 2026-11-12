"""Run the real EmbodiedAgent inside the local web server (the web counterpart of gui/app.py).

The agent is built from config.yaml with every sensor forced to simulation (the dashboard never opens a webcam) and
with the memory directory and actuator output files resolved against the repository root, as when main.py is run
from there. Its LLM calls go to the Ollama api_base in config.yaml (localhost). Local app only.
"""
import copy
import json
import threading
import time

from web.introspect import ROOT, load_config

_lock = threading.Lock()
_agent = None
_platform = None
_messages = []  # chat transcript: {'t', 'from', 'text', ...}


def _short(v, n=160):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def build_config(platform: str):
    cfg = load_config()
    if platform not in cfg["hardware"]["sensors"] and platform not in cfg["hardware"]["actuators"]:
        raise ValueError(f"unknown platform {platform!r}")
    cfg = copy.deepcopy(cfg)
    cfg["hardware"]["platform"] = platform
    for sensors in cfg["hardware"]["sensors"].values():
        for s in sensors or []:
            s["simulate"] = True
    for acts in cfg["hardware"]["actuators"].values():
        for a in acts or []:
            if a.get("output_file"):
                a["output_file"] = str(ROOT / a["output_file"])
    mem = cfg.setdefault("memory", {})
    mem["memory_dir"] = str(ROOT / mem.get("memory_dir", "memory"))
    return cfg


def _make_agent(cfg):  # separate so tests can substitute a fake
    from core.agent import EmbodiedAgent
    return EmbodiedAgent(cfg)


def start(platform: str = "mac"):
    global _agent, _platform
    with _lock:
        if _agent is not None:
            return state()
        cfg = build_config(platform)
        agent = _make_agent(cfg)
        agent.consciousness.register_message_callback(
            "sent", lambda m: _messages.append({"t": time.time(), "from": "agent", "text": m.get("content", ""),
                                                "fallback": bool(m.get("fallback")), "initiative": bool(m.get("initiative"))}))
        agent.start()
        _agent, _platform = agent, platform
    return state()


def stop():
    global _agent
    with _lock:
        if _agent is not None:
            _agent.stop()
            _agent = None
    return state()


def message(text: str):
    if _agent is None:
        raise RuntimeError("the agent is not running")
    _messages.append({"t": time.time(), "from": "human", "text": text})
    _agent.consciousness.receive_message({"id": f"msg_{int(time.time() * 1000)}", "sender": "human",
                                          "content": text, "timestamp": time.time()})
    return {"ok": True}


def action(act: dict):
    """Execute an action as ConsciousnessModule._execute_action would (memory + actuators)."""
    if _agent is None:
        raise RuntimeError("the agent is not running")
    _agent.consciousness._execute_action(act)
    return {"ok": True}


def state():
    a = _agent
    if a is None:
        return {"running": False, "messages": _messages[-50:]}
    cog = a.cognition.get_internal_state()
    wm = [{"type": w.get("type"), "importance": w.get("importance"), "content": _short(w.get("content", ""))}
          for w in a.memory.get_working_memory()]
    motors = {n: dict(act.state) for n, act in a.actuators.actuators.items() if hasattr(act, "state")}
    out = {
        "running": True, "platform": _platform,
        "monologue": a.consciousness.get_internal_monologue()[-30:],
        "emotions": cog.get("emotions", {}), "drives": cog.get("drives", {}),
        "goals": cog.get("current_goals", []), "plan": cog.get("current_plan"),
        "working_memory": wm, "memory_stats": a.memory.get_memory_stats(),
        "sensors": [[n, s.type, s.simulate] for n, s in a.sensors.sensors.items()],
        "actuators": [[n, x.type] for n, x in a.actuators.actuators.items()], "motors": motors,
        "messages": _messages[-50:],
    }
    if a.autonomy:
        out.update(topics=a.autonomy.get_topics_of_interest(), questions=a.autonomy.get_current_questions()[-10:],
                   thinking_mode=a.autonomy.get_current_thinking_mode())
    try:
        out["llm"] = a.llm_manager.get_stats()
    except Exception:  # noqa: BLE001
        out["llm"] = None
    return json.loads(json.dumps(out, default=str))
