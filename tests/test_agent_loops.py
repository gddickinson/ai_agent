"""EmbodiedAgent.start(): the perception, cognition and consciousness loops keep running (fake LLM, no network)."""
import time

import pytest

import core.agent as agent_mod
from utils.config import load_config


class FakeLLM:
    def __init__(self, config):
        self.models = {}
        self.prompts = []

    def submit_task(self, model_name, prompt, callback=None, **kw):
        self.prompts.append(model_name)

    def submit_task_sync(self, *a, **kw):
        return None

    def get_stats(self):
        return {}

    def stop(self):
        pass


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "LLMManager", FakeLLM)
    cfg = load_config("config.yaml")
    cfg["hardware"]["platform"] = "robot"
    cfg["memory"]["memory_dir"] = str(tmp_path / "mem")
    a = agent_mod.EmbodiedAgent(cfg)
    a.start()
    yield a
    a.stop()


def test_processing_loops_run_after_start(agent):
    time.sleep(1.5)
    alive = {t.name: t.is_alive() for t in agent._threads}
    assert alive == {"perception_thread": True, "cognition_thread": True, "consciousness_thread": True}
    assert agent.memory.get_recent_perceptions(limit=5), "perception loop stored nothing"
    assert "emotion_engine" in agent.llm_manager.prompts, "cognition loop never ran"
