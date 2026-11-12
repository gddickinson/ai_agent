"""Sensors/actuators without a real-hardware implementation fall back to their simulation thread (and it runs)."""
import pytest

from hardware.actuators import DisplayActuator, GenericActuator, MotorActuator, SpeakerActuator
from hardware.sensors import GenericSensor, MicrophoneSensor

CASES = [
    (MicrophoneSensor, {'name': 'mic', 'type': 'microphone', 'capture_interval': 0.05}),
    (GenericSensor, {'name': 'temp', 'type': 'temperature', 'read_interval': 0.05}),
    (DisplayActuator, {'name': 'screen', 'type': 'display'}),
    (SpeakerActuator, {'name': 'speaker', 'type': 'speaker'}),
    (MotorActuator, {'name': 'wheel', 'type': 'motor'}),
    (GenericActuator, {'name': 'arm', 'type': 'gripper'}),
]


@pytest.mark.parametrize("cls,config", CASES, ids=[c[0].__name__ for c in CASES])
def test_real_hardware_request_falls_back_to_running_simulation(cls, config):
    dev = cls({**config, 'simulate': False})
    dev.start()
    try:
        assert dev.simulate is True
        assert dev.thread is not None and dev.thread.is_alive(), "simulation thread was never started"
    finally:
        dev.stop()


def test_microphone_emits_its_simulate_data_in_order():
    """Regression: the thread used its own default phrase list instead of .simulate_data."""
    import random as _r
    mic = MicrophoneSensor({'name': 'mic', 'type': 'microphone', 'capture_interval': 0.0})
    mic.running = True
    orig = _r.random
    _r.random = lambda: 0.0  # always emit
    try:
        import threading, time
        t = threading.Thread(target=mic._simulate_microphone, daemon=True)
        t.start()
        time.sleep(0.05)
        mic.running = False
        t.join(1)
    finally:
        _r.random = orig
    texts = [mic.data_queue.get_nowait()['text'] for _ in range(3)]
    assert texts == mic.simulate_data[:3]
