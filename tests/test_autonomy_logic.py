"""Non-LLM autonomy logic: topic-interest updates and conversation-starter selection (fake LLM, no network)."""
import random

import pytest

from core.autonomy import AutonomyModule


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def submit_task_sync(self, model_name, prompt, **kw):
        self.prompts.append(prompt)
        return {'text': 'Shall we talk?'}


class FakeMemory:
    def retrieve_episodic_memories(self, limit=10, **kw):
        return []


def module():
    return AutonomyModule({}, FakeLLM(), FakeMemory(), None)


def test_new_topic_with_spaces_is_not_decayed_in_the_same_update():
    a = module()
    a.topics_of_interest = {'learning': 0.9}
    a._update_topics_of_interest(['Machine Learning', 'learning'])
    assert a.topics_of_interest['machine_learning'] == 0.5
    assert a.topics_of_interest['learning'] == pytest.approx(1.0)


def test_existing_topic_named_with_spaces_gets_the_boost_only():
    a = module()
    a.topics_of_interest = {'self_awareness': 0.5, 'emotions': 0.6}
    a._update_topics_of_interest(['Self Awareness'])
    assert a.topics_of_interest['self_awareness'] == pytest.approx(0.6)
    assert a.topics_of_interest['emotions'] == pytest.approx(0.6 * 0.99)


def test_topic_starter_uses_the_topic_prompt(monkeypatch):
    a = module()
    a.topics_of_interest = {'memory_systems': 0.9}
    monkeypatch.setattr(random, 'choice', lambda seq: seq[0])  # 'topic', then the first high-interest topic
    msg = a.initiate_conversation()
    assert msg['content'] == 'Shall we talk?' and msg['initiative'] is True
    assert 'about the topic: memory systems' in a.llm_manager.prompts[0]


def test_question_starter_and_reflection_fallback(monkeypatch):
    a = module()
    a.questions = ['What is memory?']
    monkeypatch.setattr(random, 'choice', lambda seq: 'question' if 'question' in seq else seq[0])
    a.initiate_conversation()
    assert 'What is memory?' in a.llm_manager.prompts[-1]
    a.questions = []
    a.initiate_conversation()
    assert 'recent reflection' in a.llm_manager.prompts[-1]


def test_thought_interval_shortens_when_idle():
    a = AutonomyModule({'thought_interval': 20.0, 'idle_threshold': 120.0}, FakeLLM(), FakeMemory(), None)
    assert a._thought_interval(10) == 20.0
    assert a._thought_interval(200) == pytest.approx(20.0 * (0.8 - 0.2))
    assert a._thought_interval(5000) == 5.0


def test_topic_weights_prefer_unexplored_topics():
    a = module()
    a.topics_of_interest = {'a': 0.8, 'b': 0.8, 'low': 0.1}
    a.topic_last_explored = {'a': 10000.0}
    w = a._topic_weights(10000.0 + 1800)  # 'a' explored 30 min ago, 'b' never
    assert set(w) == {'a', 'b'} and sum(w.values()) == pytest.approx(1.0)
    assert w['b'] / w['a'] == pytest.approx(2.0 / 1.5)
