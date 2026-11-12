// JS port of hardware/motor_model.py and hardware/actuators.py (ActuatorManager routing + the simulated actuators).
// Held to the Python results by actuators_engine.test.mjs (fixture: tests/fixtures/engine_fixture.json).

const get = (o, k, d) => (o && Object.prototype.hasOwnProperty.call(o, k) ? o[k] : d);

// ---- motor_model.py --------------------------------------------------------
export const TICK = 0.1;
export const newMotorState = () => ({ speed: 0, position: 0, target: null });

export function applyCommand(state, command) {
  const s = { ...state };
  const cmd = get(command, 'command', 'stop');
  if (cmd === 'stop') { s.speed = 0; s.target = null; }
  else if (cmd === 'move') {
    const speed = get(command, 'speed', 0.5), dir = get(command, 'direction', 'forward');
    s.speed = dir === 'forward' ? speed : dir === 'backward' ? -speed : 0;
    s.target = null;
  } else if (cmd === 'position') {
    const target = get(command, 'position', s.position), speed = get(command, 'speed', 0.5);
    s.target = target;
    s.speed = target > s.position ? speed : target < s.position ? -speed : 0;
  }
  return s;
}

export function step(state) {
  const s = { ...state };
  if (s.speed !== 0) {
    s.position += s.speed * TICK;
    const t = s.target;
    if (t !== null && ((s.speed > 0 && s.position >= t) || (s.speed < 0 && s.position <= t))) {
      s.position = t; s.speed = 0; s.target = null;
    }
  }
  return s;
}

// ---- simulated actuators ------------------------------------------------------
const CLASS = { display: 'DisplayActuator', speaker: 'SpeakerActuator', motor: 'MotorActuator' };

class Actuator {
  constructor(cfg) {
    this.config = cfg; this.name = get(cfg, 'name', 'unknown'); this.type = get(cfg, 'type', 'unknown');
    this.cls = CLASS[this.type] || 'GenericActuator';
    this.queue = [];      // commands waiting for the simulation thread
    this.output = [];     // what the simulation produced (display text, speech, log lines)
    if (this.type === 'motor') { this.motorType = get(cfg, 'motor_type', 'generic'); this.state = newMotorState(); }
  }
  execute(command) {
    if (this.type === 'motor') command.priority = get(command, 'priority', 50);
    this.queue.push(command);
    return true;
  }
  // One pass of the simulation thread (Python sleeps 0.1 s per motor loop; display/speaker drain their queue).
  tick(time) {
    if (this.type === 'motor') {
      const cmd = this.queue.shift();
      if (cmd) { this.state = applyCommand(this.state, cmd); this.output.push({ time, text: describe(cmd) }); }
      this.state = step(this.state);
      return;
    }
    while (this.queue.length) {
      const c = this.queue.shift();
      if (this.type === 'display') this.output.push({ time, text: String(get(c, 'content', '')), kind: get(c, 'content_type', 'text') });
      else if (this.type === 'speaker') this.output.push({ time, text: String(get(c, 'content', '')), kind: 'speech' });
      else this.output.push({ time, text: `COMMAND: ${JSON.stringify(c)}` });
    }
    if (this.output.length > 200) this.output.splice(0, this.output.length - 200);
  }
}

function describe(c) {
  const k = get(c, 'command', 'stop');
  if (k === 'move') return `moving ${get(c, 'direction', 'forward')} at speed ${get(c, 'speed', 0.5)}`;
  if (k === 'position') return `moving to position ${get(c, 'position', '?')}`;
  return k === 'stop' ? 'stopped' : `ignored command '${k}'`;
}

// ---- ActuatorManager -----------------------------------------------------------
export class ActuatorManager {
  constructor(config, platform) {
    this.platform = platform; this.actuators = {};
    for (const ac of get(config, platform, get(config, 'default', [])) || []) {
      const type = get(ac, 'type', 'generic');
      const name = get(ac, 'name', `${type}_${Object.keys(this.actuators).length}`);
      if (!get(ac, 'enabled', true)) continue;
      this.actuators[name] = new Actuator(ac);
    }
  }
  byType(t) { return Object.keys(this.actuators).filter(n => this.actuators[n].type === t); }

  executeAction(actionType, target, parameters = {}) {
    let success = false;
    const each = (type, payload) => {
      const names = this.byType(type);
      if (!names.length) return false;
      for (const n of names) {
        if (target === 'all' || target === n) { this.actuators[n].execute(payload()); success = true; }
      }
      return success;
    };
    if (actionType === 'move') {
      return each('motor', () => ({ command: 'move', direction: get(parameters, 'direction', 'forward'),
        speed: get(parameters, 'speed', 0.5) }));
    }
    if (actionType === 'speak') {
      return each('speaker', () => ({ content: get(parameters, 'text', ''), volume: get(parameters, 'volume', 1.0),
        rate: get(parameters, 'rate', 1.0) }));
    }
    if (actionType === 'display') {
      return each('display', () => ({ content: get(parameters, 'text', ''),
        content_type: get(parameters, 'content_type', 'text') }));
    }
    const a = this.actuators[target];
    return a ? a.execute({ command: actionType, ...parameters }) : false;
  }

  communicate(message) {
    const content = get(message, 'content', '');
    if (!content) return false;
    let success = false;
    for (const n of this.byType('display')) {
      success = this.actuators[n].execute({ content, content_type: 'text', message_id: get(message, 'id', null) }) || success;
    }
    if (get(message, 'speech_output', true)) {
      for (const n of this.byType('speaker')) {
        success = this.actuators[n].execute({ content, volume: get(message, 'volume', 1.0), rate: get(message, 'rate', 1.0) }) || success;
      }
    }
    return success;
  }

  tick(time) { for (const a of Object.values(this.actuators)) a.tick(time); }
}
