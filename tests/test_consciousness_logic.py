"""Consciousness module logic that needs no LLM (fake LLM/memory/actuators)."""
import time

from core.consciousness import ConsciousnessModule


class Fake:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*a, **kw):
            self.calls.append((name, a, kw))
            return [] if name.startswith('get') else None
        return record


def module():
    c = ConsciousnessModule({}, Fake(), Fake(), Fake())
    c.autonomy = object()  # autonomy present: no LLM monologue generation in process()
    return c


def test_initiation_waits_for_the_startup_delay():
    c = module()
    c.process()
    assert c.can_initiate_conversation is False  # used to switch on at the first cycle
    c.start_time = time.time() - c.initiative_startup_delay - 1
    c.process()
    assert c.can_initiate_conversation is True
