"""Pure model of a simulated motor: the state update used by MotorActuator's simulation thread.

No threads, no I/O, so it can be unit-tested and mirrored by the web simulator (web/static/actuators_engine.js).
A state is a dict {'speed', 'position', 'target'}; the simulation thread applies each queued command with
apply_command() and then calls step() every TICK seconds.

Commands (as queued by ActuatorManager.execute_action):
    {'command': 'stop'}                                       speed -> 0
    {'command': 'move', 'direction': 'forward'|'backward', 'speed': 0..1}
                                                              runs continuously until the next command
    {'command': 'position', 'position': p, 'speed': 0..1}     drives towards p and stops there
"""
from typing import Any, Dict

TICK = 0.1  # seconds between simulation steps; position advances by speed * TICK per step


def new_state() -> Dict[str, Any]:
    """A stopped motor at position 0 with no position target."""
    return {'speed': 0.0, 'position': 0.0, 'target': None}


def apply_command(state: Dict[str, Any], command: Dict[str, Any]) -> Dict[str, Any]:
    """Return the state after receiving `command` (unknown commands leave it unchanged)."""
    s = dict(state)
    cmd = command.get('command', 'stop')
    if cmd == 'stop':
        s['speed'] = 0.0
        s['target'] = None
    elif cmd == 'move':
        speed = command.get('speed', 0.5)
        direction = command.get('direction', 'forward')
        s['speed'] = speed if direction == 'forward' else -speed if direction == 'backward' else 0.0
        s['target'] = None  # continuous motion: no position to stop at
    elif cmd == 'position':
        target = command.get('position', s['position'])
        speed = command.get('speed', 0.5)
        s['target'] = target
        s['speed'] = speed if target > s['position'] else -speed if target < s['position'] else 0.0
    return s


def step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Advance one TICK: integrate position, stopping at the target of a 'position' command."""
    s = dict(state)
    if s['speed'] != 0.0:
        s['position'] += s['speed'] * TICK
        t = s['target']
        if t is not None and ((s['speed'] > 0 and s['position'] >= t) or (s['speed'] < 0 and s['position'] <= t)):
            s['position'] = t
            s['speed'] = 0.0
            s['target'] = None
    return s
