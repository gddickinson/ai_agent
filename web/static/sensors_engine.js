// JS port of the simulated sensors in hardware/sensors.py and SensorManager.get_readings().
// Only what the Python code simulates: a camera that picks one of its description strings, a microphone that emits
// its phrases in rotation (30% chance per capture interval), and generic sensors (lidar, temperature, battery, ...)
// that report a uniform random value between simulate_min and simulate_max. Nothing here models the room: the
// repo's lidar is a random number too. Time is simulated seconds; rng() is a seeded uniform [0, 1) generator.

const get = (o, k, d) => (o && Object.prototype.hasOwnProperty.call(o, k) ? o[k] : d);

export const DEFAULT_CAMERA_DATA = ['A desk with a computer and some papers.',
  'A person sitting at a desk, working on a computer.', 'An empty room with white walls and a window.'];
export const DEFAULT_MIC_DATA = ['Hello, can you hear me?', "What's the weather like today?",
  'I need some help with a problem.', 'This is a test of the audio system.', ''];

class Camera {
  constructor(cfg, rng) {
    this.cls = 'CameraSensor'; this.type = 'camera'; this.rng = rng;
    this.name = get(cfg, 'name', 'camera');
    this.interval = get(cfg, 'frame_interval', 1.0);
    this.simulate = get(cfg, 'simulate', false);
    this.realRequested = !this.simulate;  // a real webcam: the browser shows the simulation fallback instead
    this.simulate = true;
    this.simulate_data = get(cfg, 'simulate_data', DEFAULT_CAMERA_DATA);
    this.last = -Infinity;
  }
  advance() {}
  getData(now) {
    if (now - this.last < this.interval) return null;
    this.last = now;
    const description = this.simulate_data[Math.floor(this.rng() * this.simulate_data.length)];
    return { type: 'camera', name: this.name, timestamp: now, simulated: true, description, frame: null };
  }
}

class Microphone {
  constructor(cfg, rng) {
    this.cls = 'MicrophoneSensor'; this.type = 'microphone'; this.rng = rng;
    this.name = get(cfg, 'name', 'unknown');
    this.interval = get(cfg, 'capture_interval', 2.0);
    this.simulate = get(cfg, 'simulate', true);
    this.realRequested = !this.simulate; this.simulate = true;  // real microphone: not implemented in the repo
    this.simulate_data = get(cfg, 'simulate_data', DEFAULT_MIC_DATA);
    this.queue = []; this.count = 0; this.next = 0;
  }
  // The simulation thread: every capture_interval, a 30% chance to queue the next phrase (queue holds 10).
  advance(now) {
    while (now >= this.next) {
      if (this.rng() < 0.3 && this.queue.length < 10) {
        const text = this.simulate_data.length ? this.simulate_data[this.count % this.simulate_data.length]
          : 'Simulated audio input.';
        this.count += 1;
        this.queue.push({ type: 'microphone', timestamp: this.next, audio_id: this.count, text, simulated: true });
      }
      this.next += this.interval;
    }
  }
  getData() { return this.queue.shift() || {}; }
}

class Generic {
  constructor(cfg, rng) {
    this.cls = 'GenericSensor'; this.rng = rng;
    this.type = get(cfg, 'type', 'generic'); this.name = get(cfg, 'name', 'unknown');
    this.interval = get(cfg, 'read_interval', 5.0);
    this.simulate = get(cfg, 'simulate', true);
    this.realRequested = !this.simulate; this.simulate = true;  // real sensors: not implemented in the repo
    this.min = get(cfg, 'simulate_min', 0); this.max = get(cfg, 'simulate_max', 100);
    this.unit = get(cfg, 'simulate_unit', '');
    this.queue = []; this.next = 0;
  }
  // Every read_interval: one uniform reading; the queue keeps the 5 newest.
  advance(now) {
    while (now >= this.next) {
      const value = this.min + (this.max - this.min) * this.rng();
      this.queue.push({ type: this.type, timestamp: this.next, value, unit: this.unit, source: this.name });
      if (this.queue.length > 5) this.queue.shift();
      this.next += this.interval;
    }
  }
  getData() { return this.queue.shift() || {}; }
}

export class SensorManager {
  constructor(config, platform, rng) {
    this.platform = platform; this.sensors = {};
    for (const sc of get(config, platform, get(config, 'default', [])) || []) {
      const type = get(sc, 'type', 'generic');
      const name = get(sc, 'name', `${type}_${Object.keys(this.sensors).length}`);
      if (!get(sc, 'enabled', true)) continue;
      this.sensors[name] = type === 'camera' ? new Camera(sc, rng) : type === 'microphone' ? new Microphone(sc, rng)
        : new Generic(sc, rng);
    }
  }
  advance(now) { for (const s of Object.values(this.sensors)) s.advance(now); }
  // {sensor_name: reading} for every sensor with a reading (SensorManager.get_readings).
  getReadings(now) {
    const out = {};
    for (const [n, s] of Object.entries(this.sensors)) {
      const d = s.getData(now);
      if (d && Object.keys(d).length) out[n] = d;
    }
    return out;
  }
}
