"""Sensor readings reach perception: one agent perception cycle stores raw perceptions, fills working memory and
queues the right interpreter for each sensor type. No LLM is called (a fake LLM manager records the tasks)."""
import time

import pytest

from core.agent import EmbodiedAgent
from core.memory import MemoryManager
from core.perception import PerceptionModule
from hardware.sensors import SensorManager

READINGS = {
    'webcam': {'type': 'camera', 'name': 'webcam', 'timestamp': 1.0, 'simulated': True,
               'description': 'A desk with a computer.', 'frame': None},
    'microphone': {'type': 'microphone', 'timestamp': 1.0, 'audio_id': 1, 'text': 'Hello, agent.', 'simulated': True},
    'temperature': {'type': 'temperature', 'timestamp': 1.0, 'value': 21.5, 'unit': 'C', 'source': 'temperature'},
}


class FakeLLM:
    def __init__(self):
        self.tasks = []

    def submit_task(self, model_name, prompt, callback=None, **kw):
        self.tasks.append((model_name, prompt))


@pytest.fixture
def agent(tmp_path):
    sensors = SensorManager({'robot': [{'name': n, 'type': r['type'], 'simulate': True} for n, r in READINGS.items()]},
                            'robot')
    for name, sensor in sensors.sensors.items():  # canned readings instead of threads/random data
        sensor.get_data = (lambda r: (lambda: dict(r)))(READINGS[name])
    a = object.__new__(EmbodiedAgent)
    a.logger = __import__('logging').getLogger('test')
    a.sensors = sensors
    a.memory = MemoryManager({'memory_dir': str(tmp_path / 'mem')})
    a.llm = FakeLLM()
    a.perception = PerceptionModule({'process_intervals': {'camera': 0, 'microphone': 0, 'temperature': 0}},
                                    a.llm, a.memory)
    yield a
    a.memory.stop()


def test_readings_are_keyed_by_sensor_name(agent):
    assert set(agent.sensors.get_readings()) == set(READINGS)


def test_perception_cycle_processes_every_sensor(agent):
    assert agent._perception_cycle() == 3
    kinds = {p['type'] for p in agent.memory.get_recent_perceptions(limit=10)}
    assert kinds == {'camera', 'microphone', 'temperature'}
    contents = [w['content'] for w in agent.memory.get_working_memory() if w['type'] == 'perception']
    assert 'Visual: A desk with a computer.' in contents and 'Audio: Hello, agent.' in contents
    models = [m for m, _ in agent.llm.tasks]
    assert models == ['audio_interpreter', 'general_interpreter']  # simulated camera needs no vision model
    assert 'Hello, agent.' in agent.llm.tasks[0][1]
    assert '21.5' in agent.llm.tasks[1][1]


def test_process_intervals_respected(agent):
    agent.perception.process_intervals = {'camera': 1000, 'microphone': 1000, 'temperature': 1000}
    agent.perception.last_processed = {k: time.time() for k in ('camera', 'microphone', 'temperature')}
    agent._perception_cycle()
    assert agent.llm.tasks == [] and agent.memory.get_recent_perceptions() == []
