"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# ---- project-specific API tests (no LLM, no hardware, no network) -------------------------------
def test_info_counts_come_from_the_code():
    i = client.get("/api/info").json()
    assert i["platforms"] == ["mac", "robot"] and i["sensors"]["robot"] == 5 and i["actuators"]["robot"] == 4
    assert i["tables"] == 8 and i["emotions"] == 7 and i["thinking_modes"] == 6


def test_defaults_match_the_modules():
    from core.autonomy import DEFAULT_TOPICS
    d = client.get("/api/defaults").json()
    assert d["topics"] == DEFAULT_TOPICS and d["emotions"]["curiosity"] == 0.5
    assert d["consciousness"]["fallback_response"].startswith("I'm sorry")


def test_architecture_lists_core_modules():
    paths = {m["path"] for m in client.get("/api/architecture").json()}
    assert {"core/agent.py", "hardware/motor_model.py", "llm/manager.py"} <= paths


def test_static_export_has_no_local_data(tmp_path):
    """Nothing from memory.db or the output logs (personal conversations) goes into the static site."""
    assert not any(p.startswith("/local") for p in STATIC_ENDPOINTS)
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    text = "".join(p.read_text(errors="ignore") for p in (tmp_path / "site").rglob("*.json"))
    assert "Lumina" not in text and "conversation_17" not in text


def test_memory_table_rejects_unknown_tables():
    assert client.get("/api/local/memory/sqlite_master").status_code == 404


def test_agent_endpoints_with_a_fake_agent(monkeypatch):
    from web import agent_control

    class Fake:
        def __init__(self, cfg):
            self.cfg, self.sent, self.started = cfg, [], False
            fake = self

            class C:
                def register_message_callback(self, kind, cb): fake.cb = cb
                def receive_message(self, m): fake.sent.append(m); fake.cb({"content": "fallback", "fallback": True})
                def get_internal_monologue(self): return ["[00:00:00] hi"]
                def _execute_action(self, a): fake.sent.append(a)
            self.consciousness = C()
            self.cognition = type("G", (), {"get_internal_state": lambda s: {"emotions": {"joy": 0.1}}})()
            self.memory = type("M", (), {"get_working_memory": lambda s: [], "get_memory_stats": lambda s: {}})()
            self.sensors = type("S", (), {"sensors": {}})()
            self.actuators = type("A", (), {"actuators": {}})()
            self.autonomy = None
            self.llm_manager = None

        def start(self): self.started = True
        def stop(self): self.started = False

    monkeypatch.setattr(agent_control, "_make_agent", Fake)
    assert client.get("/api/local/agent").json()["running"] is False
    assert client.post("/api/local/agent/message", json={"text": "x"}).status_code == 409
    s = client.post("/api/local/agent/start", json={"platform": "robot"}).json()
    assert s["running"] and s["platform"] == "robot"
    cfg = agent_control._agent.cfg
    assert all(x["simulate"] for xs in cfg["hardware"]["sensors"].values() for x in xs or [])
    assert client.post("/api/local/agent/message", json={"text": "hello"}).json() == {"ok": True}
    assert client.post("/api/local/agent/action", json={"type": "move", "target": "all"}).status_code == 200
    msgs = client.get("/api/local/agent").json()["messages"]
    assert [m["from"] for m in msgs[-2:]] == ["human", "agent"]
    assert client.post("/api/local/agent/stop").json()["running"] is False
    assert client.post("/api/local/agent/start", json={"platform": "mars"}).status_code == 400
