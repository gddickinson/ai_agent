// Sensors & actuators: each platform's devices from config.yaml, what the code does for real vs in simulation, and how
// ActuatorManager routes actions (the routing table is computed by the node-tested JS port).
import { api, app, esc, route } from './core.js';
import { ActuatorManager } from './actuators_engine.js';

const SENSOR_NOTES = {
  camera: ['CameraSensor', 'Real: OpenCV <code>VideoCapture</code> on <code>device</code> (then <code>alternative_devices</code>), frames resized to 512×384 and sent base64 to the vision model. Falls back to simulation if OpenCV is missing or no device opens.',
    'Simulated: one of its <code>simulate_data</code> descriptions at random, at most once per <code>frame_interval</code>.'],
  microphone: ['MicrophoneSensor', 'Real: not implemented (falls back to simulation).',
    'Simulated: every <code>capture_interval</code>, a 30% chance to emit the next of its phrases in rotation (pre-transcribed text).'],
  generic: ['GenericSensor', 'Real: not implemented (falls back to simulation).',
    'Simulated: every <code>read_interval</code>, a uniform random value between <code>simulate_min</code> and <code>simulate_max</code> (defaults 0 to 100). A simulated lidar is one such number, not a scan.'],
};
const ACT_NOTES = {
  display: ['DisplayActuator', 'Simulated: logs the text and appends it, timestamped, to <code>output_file</code> if set.'],
  speaker: ['SpeakerActuator', 'Simulated: logs “SPEECH: …” and appends it to <code>output_file</code> if set. No audio is played.'],
  motor: ['MotorActuator', 'Simulated (hardware/motor_model.py): every 0.1 s, position += speed × 0.1; <code>move</code> runs until the next command, <code>position</code> stops at its target, <code>stop</code> halts.'],
  generic: ['GenericActuator', 'Simulated: logs each command and waits 0.2 s.'],
};

function routing(cfg, platform) {
  const tests = [['move', 'all', { direction: 'forward' }], ['move', 'left_wheel', {}], ['speak', 'all', { text: '…' }],
    ['display', 'screen', { text: '…' }], ['position', 'right_wheel', { position: 1 }], ['observe', 'room', {}]];
  return tests.map(([t, target, p]) => {
    const am = new ActuatorManager({ [platform]: cfg.actuators[platform] }, platform);
    const ok = am.executeAction(t, target, p);
    const to = Object.entries(am.actuators).filter(([, a]) => a.queue.length).map(([n]) => n);
    return `<tr><td class="mono">${t}</td><td class="mono">${target}</td><td class="${ok ? 'ok' : 'err'}">${ok}</td><td>${to.map(esc).join(', ') || '–'}</td></tr>`;
  }).join('');
}

route('hardware', async () => {
  const cfg = await api('/config');
  const plat = p => {
    const sens = cfg.sensors[p].map(s => {
      const n = SENSOR_NOTES[s.type] || SENSOR_NOTES.generic;
      const iv = s.frame_interval ?? s.capture_interval ?? s.read_interval;
      const range = s.simulate_min !== undefined ? ` · ${s.simulate_min} to ${s.simulate_max} ${esc(s.simulate_unit || '')}` : '';
      return `<tr><td><b>${esc(s.name)}</b><br><small class="mute">${esc(s.type)} · ${n[0]}</small></td>
        <td class="mono">${iv ?? '–'}s</td><td>${s.simulate ? '<span class="tag a">simulated</span>' : '<span class="tag b">real requested</span>'}${range}</td>
        <td class="small">${s.simulate ? n[2] : n[1]}</td></tr>`;
    }).join('');
    const acts = cfg.actuators[p].map(a => {
      const n = ACT_NOTES[a.type] || ACT_NOTES.generic;
      return `<tr><td><b>${esc(a.name)}</b><br><small class="mute">${esc(a.type)}${a.motor_type ? ' · ' + esc(a.motor_type) : ''} · ${n[0]}</small></td>
        <td class="small">${n[1]}${a.output_file ? ` → <code>${esc(a.output_file)}</code>` : ''}</td></tr>`;
    }).join('');
    return `<h2>${esc(p)} platform${p === cfg.platform ? ' <span class="tag a">default in config.yaml</span>' : ''}</h2>
      <h3>Sensors</h3><div class="scroll"><table><tr><th>sensor</th><th>every</th><th>mode</th><th>what the code does</th></tr>${sens}</table></div>
      <h3>Actuators</h3><div class="scroll"><table><tr><th>actuator</th><th>behaviour</th></tr>${acts}</table></div>
      <h3>Action routing</h3><div class="scroll"><table><tr><th>action</th><th>target</th><th>success</th><th>queued on</th></tr>${routing(cfg, p)}</table></div>`;
  };
  app().innerHTML = `
  <p class="kicker">Architecture · hardware/</p><h1>Sensors &amp; actuators</h1>
  <p class="lead"><code>SensorManager</code> and <code>ActuatorManager</code> build one object per device listed for the active
    platform in config.yaml (cameras, microphones and motors get their own classes; anything else is generic) and skip
    disabled ones. Of the real devices, only the camera is implemented; every other “real” device logs a warning and runs its
    simulation instead. <code>hardware/adapters/mac_adapters.py</code> and <code>robot_adapters.py</code> are empty placeholders.</p>
  <div class="callout">Actions arrive as <code>execute_action(type, target, parameters)</code>: <b>move</b>, <b>speak</b> and
    <b>display</b> go to every actuator of that kind when the target is <code>all</code>, or to the one named; any other action
    type is sent as a command to the actuator named by the target. The consciousness module also
    <code>communicate()</code>s every reply to all displays and speakers.</div>
  ${cfg.platforms.map(plat).join('')}
  <p><a class="btn" href="#/world?platform=robot">Drive the robot in the simulated world</a></p>`;
});
