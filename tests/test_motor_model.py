"""Simulated motor: 'move' runs continuously, 'position' stops at its target, 'stop' halts (hardware/motor_model.py)."""
import time

import pytest

from hardware import motor_model as mm
from hardware.actuators import MotorActuator


def run(cmds_and_ticks):
    s = mm.new_state()
    for cmd, ticks in cmds_and_ticks:
        if cmd:
            s = mm.apply_command(s, cmd)
        for _ in range(ticks):
            s = mm.step(s)
    return s


def test_move_forward_keeps_moving():
    s = run([({'command': 'move', 'direction': 'forward', 'speed': 0.5}, 10)])
    assert s['position'] == pytest.approx(0.5) and s['speed'] == 0.5


def test_move_backward():
    s = run([({'command': 'move', 'direction': 'backward', 'speed': 1.0}, 3)])
    assert s['position'] == pytest.approx(-0.3)


def test_position_stops_at_target():
    s = run([({'command': 'position', 'position': 0.25, 'speed': 1.0}, 10)])
    assert s['position'] == 0.25 and s['speed'] == 0.0 and s['target'] is None


def test_stop():
    s = run([({'command': 'move', 'speed': 1.0}, 2), ({'command': 'stop'}, 5)])
    assert s['position'] == pytest.approx(0.2) and s['speed'] == 0.0


def test_move_after_position_is_continuous():
    s = run([({'command': 'position', 'position': -0.1, 'speed': 1.0}, 5), ({'command': 'move', 'speed': 1.0}, 4)])
    assert s['position'] == pytest.approx(0.3)


def test_simulated_motor_thread_moves():
    """Regression: the thread used to snap a 'move' back to target 0 and stop after one tick."""
    m = MotorActuator({'name': 'left_wheel', 'type': 'motor', 'simulate': True})
    m.start()
    try:
        m.execute({'command': 'move', 'direction': 'forward', 'speed': 1.0})
        time.sleep(0.45)
        assert m.state['position'] > 0.15 and m.state['speed'] == 1.0
    finally:
        m.stop()
