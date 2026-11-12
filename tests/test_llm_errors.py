"""When the Ollama server is unreachable the LLM manager reports an error (callback(None, task)), so the agent
answers with its fallback response instead of speaking the error text. requests.post is patched: no network."""
import time

import pytest
import requests

from core.consciousness import ConsciousnessModule
from llm.manager import LLMManager

CONFIG = {"num_workers": 1, "models": {"consciousness_engine": {"type": "ollama", "model_id": "llama3:latest",
                                                                 "api_base": "http://localhost:11434/api"}}}


@pytest.fixture
def down(monkeypatch):
    def refuse(*a, **kw):
        raise requests.exceptions.ConnectionError("Connection refused")
    monkeypatch.setattr(requests, "post", refuse)


@pytest.fixture
def llm(down):
    m = LLMManager(CONFIG)
    yield m
    m.stop()


def test_unreachable_server_is_an_error_not_a_reply(llm):
    got = {}
    llm.submit_task("consciousness_engine", "hi", callback=lambda r, t: got.update(result=r, error=t.error))
    for _ in range(50):
        if got:
            break
        time.sleep(0.05)
    assert got["result"] is None and "Connection refused" in got["error"]
    assert llm.submit_task_sync("consciousness_engine", "hi", timeout=3) is None


class Rec:
    def __getattr__(self, name):
        return lambda *a, **kw: [] if name.startswith("get") else None


def test_agent_sends_its_fallback_when_ollama_is_down(llm):
    c = ConsciousnessModule({}, llm, Rec(), Rec())
    sent = []
    c.register_message_callback("sent", sent.append)
    c.receive_message({"id": "m1", "sender": "human", "content": "Hello"})
    for _ in range(60):
        if sent:
            break
        time.sleep(0.05)
    assert sent and sent[0].get("fallback") is True
    assert "Error processing request" not in sent[0]["content"]
