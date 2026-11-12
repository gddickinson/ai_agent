// node web/static/actuators_engine.test.mjs: the JS motor model and ActuatorManager reproduce hardware/motor_model.py
// and hardware/actuators.py (fixture: tests/fixtures/engine_fixture.json, from scripts/make_engine_fixture.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TICK, newMotorState, applyCommand, step, ActuatorManager } from './actuators_engine.js';
import { SensorManager } from './sensors_engine.js';
import { rng } from './core_rng.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/engine_fixture.json', import.meta.url)));
const close = (a, b, what) => assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${what}: js ${a} vs py ${b}`);

assert.equal(TICK, fx.motor.tick);
let n = 0;
for (const [name, c] of Object.entries(fx.motor.cases)) {
  let s = newMotorState(), k = 0;
  for (const [cmd, ticks] of c.steps) {
    s = applyCommand(s, cmd);
    for (let i = 0; i < ticks; i++, k++) {
      s = step(s);
      const [speed, pos, target] = c.states[k];
      close(s.speed, speed, `${name} tick ${k} speed`); close(s.position, pos, `${name} tick ${k} position`);
      assert.equal(s.target === null, target === null, `${name} tick ${k} target`);
      if (target !== null) close(s.target, target, `${name} target`);
      n++;
    }
  }
}

// The fixture was built from config.yaml; rebuild the same actuator config here.
const cfgActs = {};
for (const [platform, p] of Object.entries(fx.actuators)) {
  const am = new ActuatorManager({ [platform]: p.actuators.map(([name, type]) => ({ name, type })) }, platform);
  assert.deepEqual(Object.entries(am.actuators).map(([nm, a]) => [nm, a.type, a.cls]), p.actuators, `${platform} actuators`);
  cfgActs[platform] = p.actuators.map(([name, type]) => ({ name, type }));
  for (const c of p.actions) {
    const m = new ActuatorManager({ [platform]: cfgActs[platform] }, platform);
    const [type, target, params] = c.action;
    assert.equal(m.executeAction(type, target, structuredClone(params)), c.success, `${platform} ${type}/${target} success`);
    const queued = Object.fromEntries(Object.entries(m.actuators).filter(([, a]) => a.queue.length).map(([nm, a]) => [nm, a.queue]));
    assert.deepEqual(queued, c.queued, `${platform} ${type}/${target} queued commands`);
    n++;
  }
  const m = new ActuatorManager({ [platform]: cfgActs[platform] }, platform);
  for (const c of p.communicate) {
    assert.equal(m.communicate(c.message), c.success, `${platform} communicate success`);
    const queued = Object.fromEntries(Object.entries(m.actuators).filter(([, a]) => a.queue.length).map(([nm, a]) => [nm, a.queue.splice(0)]));
    assert.deepEqual(queued, c.queued, `${platform} communicate ${c.message.id}`);
    n++;
  }
}

// Sensors: the same classes, intervals, ranges and phrases as SensorManager builds from config.yaml.
for (const [platform, rows] of Object.entries(fx.sensors)) {
  const cfg = rows.map(r => ({ name: r.name, type: r.type, ...(r.class === 'CameraSensor' ? { frame_interval: r.interval }
    : r.class === 'MicrophoneSensor' ? { capture_interval: r.interval } : { read_interval: r.interval, simulate_min: r.min,
      simulate_max: r.max, simulate_unit: r.unit }), simulate_data: r.simulate_data ?? undefined }));
  for (const c of cfg) if (c.simulate_data === undefined) delete c.simulate_data;
  const sm = new SensorManager({ [platform]: cfg }, platform, rng(1));
  for (const r of rows) {
    const s = sm.sensors[r.name];
    assert.equal(s.cls, r.class, `${r.name} class`); assert.equal(s.interval, r.interval, `${r.name} interval`);
    if (r.simulate_data) assert.deepEqual(s.simulate_data, r.simulate_data, `${r.name} phrases`);
    if (r.class === 'GenericSensor') { assert.equal(s.min, r.min); assert.equal(s.max, r.max); assert.equal(s.unit, r.unit); }
    n++;
  }
  // readings stay inside their ranges and the microphone rotates through its phrases
  sm.advance(600);
  for (const s of Object.values(sm.sensors)) {
    if (s.cls === 'GenericSensor') for (const q of s.queue) assert.ok(q.value >= s.min && q.value <= s.max);
    if (s.cls === 'MicrophoneSensor') s.queue.forEach((q, i) => assert.equal(q.text, s.simulate_data[i % s.simulate_data.length]));
  }
}
console.log(`actuators/sensors engine: ${n} checks match the Python code`);
