"""Write tests/fixtures/engine_fixture.json: results of the framework's non-LLM logic, computed by the Python code,
which the JS ports in web/static/*_engine.js must reproduce (node tests: web/static/*.test.mjs).

Covered: the simulated motor model, ActuatorManager action routing and communicate(), SensorManager construction
from config.yaml, working-memory insertion/trimming, topic-interest updates and selection weights, thinking-mode
selection, the idle thought interval, emotion parsing, conversation-initiation probability, and the regex fact/name
extraction in ConsciousnessModule.receive_message. No LLM is called (stand-in objects record what would be sent).

Run: python scripts/make_engine_fixture.py   (tests/test_engine_fixture.py checks the committed file is current)
"""
import json
import logging
import queue
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

from core.autonomy import DEFAULT_TOPICS, AutonomyModule  # noqa: E402
from core.cognition import CognitionModule  # noqa: E402
from core.consciousness import ConsciousnessModule  # noqa: E402
from core.memory import MemoryManager  # noqa: E402
from hardware import motor_model  # noqa: E402
from hardware.actuators import ActuatorManager  # noqa: E402
from hardware.sensors import SensorManager  # noqa: E402
from web.introspect import load_config  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_fixture.json"

MOTOR_CASES = {
    "forward_then_stop": [[{"command": "move", "direction": "forward", "speed": 0.5}, 6], [{"command": "stop"}, 3]],
    "backward": [[{"command": "move", "direction": "backward", "speed": 0.8}, 5]],
    "sideways_is_zero": [[{"command": "move", "direction": "left", "speed": 0.8}, 3]],
    "position_overshoot": [[{"command": "position", "position": 0.33, "speed": 0.7}, 8]],
    "position_back": [[{"command": "move", "speed": 1.0}, 4], [{"command": "position", "position": -0.15, "speed": 0.4}, 20]],
    "default_speed": [[{"command": "move"}, 4]],
}

ACTIONS = [
    ["move", "all", {"direction": "forward", "speed": 0.6}],
    ["move", "left_wheel", {"direction": "backward"}],
    ["move", "forward", {"speed": 0.3}],
    ["speak", "all", {"text": "Hello there", "volume": 0.5}],
    ["speak", "speakers", {"text": "On the mac speaker"}],
    ["display", "all", {"text": "Status: OK"}],
    ["display", "screen", {"text": "Only the screen", "content_type": "markdown"}],
    ["position", "right_wheel", {"position": 2.0, "speed": 0.4}],
    ["observe", "room", {}],
]

MESSAGES = [
    "Hello, my name is Alice and I am 34 years old.",
    "I'm from Glasgow, Scotland. I like hiking and tea",
    "my age is 51",
    "I hate rain",
    "I don't like loud noise. Peter says hi",
    "What do you see?",
    "I'm in London",
    "My name is bob",
]

EMOTION_TEXTS = [
    '{"joy": 0.7, "sadness": 0.1, "anger": 0, "fear": 0.2, "surprise": 0.4, "curiosity": 0.9, "confusion": 0.1}',
    'Here is my state: {"joy": 1.5, "curiosity": "0.3", "fear": -2, "anger": "high"} hope that helps',
    '{"joy": 0.2}',
]

THINKING_EMOTIONS = [
    {"curiosity": 0.7}, {"curiosity": 0.6, "confusion": 0.65}, {"joy": 0.9}, {"sadness": 0.45},
    {"fear": 0.5, "joy": 0.61}, {"confusion": 0.2}, {"curiosity": 0.61, "joy": 0.99},
]


class FakeLLM:
    models = {"consciousness_engine": {}}

    def submit_task(self, *a, **kw):
        return None

    def submit_task_sync(self, *a, **kw):
        return None


class RecordingMemory:
    """Records store_fact/store_user_info; everything else is a no-op."""

    def __init__(self, working=None):
        self.facts, self.users, self.working = [], [], working or []

    def store_fact(self, entity, attribute, value, source="conversation"):
        self.facts.append([entity, attribute, value])

    def store_user_info(self, user_id, user_name, metadata=None):
        self.users.append([user_id, user_name])

    def get_user_info(self, user_id=None, user_name=None):
        return None

    def get_working_memory(self):
        return self.working

    def __getattr__(self, name):
        return lambda *a, **kw: None


def motors():
    out = {}
    for name, steps in MOTOR_CASES.items():
        s, rows = motor_model.new_state(), []
        for cmd, ticks in steps:
            s = motor_model.apply_command(s, cmd)
            for _ in range(ticks):
                s = motor_model.step(s)
                rows.append([s["speed"], s["position"], s["target"]])
        out[name] = {"steps": steps, "states": rows}
    return out


def drain(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def actuators(cfg):
    out = {}
    for platform in ("mac", "robot"):
        cases = []
        for action_type, target, params in ACTIONS:
            am = ActuatorManager(cfg["hardware"]["actuators"], platform)
            ok = am.execute_action(action_type, target, dict(params))
            queued = {n: drain(getattr(a, "command_queue", None) or a.output_queue) for n, a in am.actuators.items()}
            cases.append({"action": [action_type, target, params], "success": ok,
                          "queued": {n: q for n, q in queued.items() if q}})
        am = ActuatorManager(cfg["hardware"]["actuators"], platform)
        comm = []
        for msg in [{"id": "r1", "content": "Hi!"}, {"id": "r2", "content": "Quiet", "speech_output": False},
                    {"id": "r3", "content": ""}]:
            ok = am.communicate(msg)
            queued = {n: drain(getattr(a, "command_queue", None) or a.output_queue) for n, a in am.actuators.items()}
            comm.append({"message": msg, "success": ok, "queued": {n: q for n, q in queued.items() if q}})
        out[platform] = {"actions": cases, "communicate": comm,
                         "actuators": [[n, a.type, type(a).__name__] for n, a in am.actuators.items()]}
    return out


def sensors(cfg):
    out = {}
    for platform in ("mac", "robot"):
        sm = SensorManager(cfg["hardware"]["sensors"], platform)
        rows = []
        for n, s in sm.sensors.items():
            interval = getattr(s, "frame_interval", None) or getattr(s, "capture_interval", None) or s.read_interval
            row = {"name": n, "type": s.type, "class": type(s).__name__, "interval": interval,
                   "simulate": s.simulate, "simulate_data": getattr(s, "simulate_data", None)}
            if type(s).__name__ == "GenericSensor":
                row.update(min=s.simulate_min, max=s.simulate_max, unit=s.simulate_unit)
            rows.append(row)
        out[platform] = rows
    return out


def working_memory():
    seq = [["a", 0.5], ["b", 0.8], ["c", 0.3], ["d", 0.8], ["e", 0.6], ["f", 0.45], ["g", 0.9], ["h", 0.2],
           ["i", 0.7], ["j", 0.5]]
    with tempfile.TemporaryDirectory() as tmp:
        mm = MemoryManager({"memory_dir": tmp, "working_memory_capacity": 5})
        snapshots = []
        for name, imp in seq:
            mm.add_to_working_memory({"type": name, "timestamp": 0}, importance=imp)
            snapshots.append([w["type"] for w in mm.working_memory])
        spilled = [json.loads(i["data"]["content"])["type"] for i in drain(mm.memory_queue)]
        mm.stop()
    return {"capacity": 5, "adds": seq, "snapshots": snapshots, "spilled": spilled}


def autonomy(cfg):
    a = AutonomyModule(cfg["autonomy"], FakeLLM(), RecordingMemory(), None)
    a.topics_of_interest = dict(DEFAULT_TOPICS)
    updates = [["Learning", "Robotics"], ["robotics", "emotions"], [], ["Machine Learning"]] + [["robotics"]] * 3
    after = []
    for u in updates:
        a._update_topics_of_interest(u)
        after.append(dict(a.topics_of_interest))
    # Long decay: untouched topics fall below 0.2 and are removed
    b = AutonomyModule({}, FakeLLM(), RecordingMemory(), None)
    b.topics_of_interest = {"x": 0.25, "y": 0.9}
    decay = []
    for _ in range(30):
        b._update_topics_of_interest(["y"])
        decay.append(dict(b.topics_of_interest))
    weights = []
    for now, explored in [[5000.0, {}], [5000.0, {"learning": 4000.0, "creativity": 0.0}],
                          [100.0, {"self_awareness": 90.0}]]:
        a.topics_of_interest = dict(DEFAULT_TOPICS)
        a.topic_last_explored = dict(explored)
        weights.append({"now": now, "explored": explored, "weights": a._topic_weights(now)})
    a.topics_of_interest = {"x": 0.1}
    weights.append({"now": 10.0, "explored": {}, "topics": {"x": 0.1}, "weights": a._topic_weights(10.0)})
    intervals = [[t, a._thought_interval(t)] for t in [0, 60, 120, 121, 300, 600, 700, 1000, 5000]]
    modes = []
    for emo in THINKING_EMOTIONS:
        m = AutonomyModule({}, FakeLLM(), RecordingMemory([{"type": "emotion_change", "content": {"current": emo}}]), None)
        m.current_thinking_mode = "predictive"
        m._select_thinking_mode()
        modes.append([emo, m.current_thinking_mode])
    return {"default_topics": DEFAULT_TOPICS, "updates": updates, "after": after, "decay_start": {"x": 0.25, "y": 0.9}, "decay": decay,
            "weights": weights, "min_interest_threshold": a.min_interest_threshold,
            "thought_interval": a.thought_interval, "idle_threshold": a.idle_threshold, "intervals": intervals,
            "modes": modes}


def cognition():
    c = CognitionModule({}, FakeLLM(), RecordingMemory())
    return {"parsed": [[t, c._parse_emotion_output(t)] for t in EMOTION_TEXTS]}


def consciousness():
    probs = [[cur, idle, hour, ConsciousnessModule._initiation_probability(cur, idle, hour)]
             for cur in (0.0, 0.3, 0.8) for idle in (0, 900, 7200, 30000) for hour in (3, 9, 22, 23)]
    msgs = []
    for text in MESSAGES:
        mem = RecordingMemory()
        c = ConsciousnessModule({}, FakeLLM(), mem, None)
        c._process_incoming_messages = lambda: None
        c.receive_message({"id": "m1", "sender": "human", "content": text})
        msgs.append({"text": text, "facts": mem.facts, "users": mem.users})
    return {"initiation": probs, "messages": msgs}


def build():
    cfg = load_config()
    return {"motor": {"tick": motor_model.TICK, "cases": motors()}, "actuators": actuators(cfg),
            "sensors": sensors(cfg), "working_memory": working_memory(), "autonomy": autonomy(cfg),
            "cognition": cognition(), "consciousness": consciousness()}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")
