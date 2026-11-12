// The agent's processing loops (core/agent.py, cognition, consciousness, autonomy) run on a simulated clock, using
// the JS ports of the non-LLM logic. Wherever the Python agent would send a prompt to an Ollama model, the simulator
// records the request and takes the repo's own no-LLM path instead:
//   message reply  -> ConsciousnessModule._generate_fallback_response (what the agent says when the LLM fails)
//   emotions       -> CognitionModule._parse_emotion_output on a reply without JSON (random drift), or frozen
//   initiative     -> AutonomyModule.initiate_conversation's generic fallback starter
//   actions        -> you supply the action JSON the planning/action LLM would have produced
import { SensorManager } from './sensors_engine.js';
import { ActuatorManager, TICK } from './actuators_engine.js';
import { WorkingMemory, Monologue, perceive, thoughtInterval, selectThinkingMode, selectTopic, driftEmotions,
  significantChange, initiationProbability, extractFacts, FALLBACK_RESPONSE } from './mind_engine.js';

const START_HOUR = 10;  // the simulated clock starts at 10:00
const FALLBACK_STARTER = "I've been thinking about consciousness. What makes an entity self-aware?";
const g = (o, k, d) => (o && o[k] !== undefined ? o[k] : d);

export function clock(t) {
  const s = Math.floor(START_HOUR * 3600 + t) % 86400;
  return [s / 3600, (s / 60) % 60, s % 60].map(v => String(Math.floor(v)).padStart(2, '0')).join(':');
}

export class AgentSim {
  constructor({ config, defaults, platform, rng, emotionMode = 'drift' }) {
    this.cfg = config; this.def = defaults; this.platform = platform; this.rng = rng; this.emotionMode = emotionMode;
    this.t = 0;
    this.sensors = new SensorManager({ [platform]: config.sensors[platform] || [] }, platform, rng);
    this.actuators = new ActuatorManager({ [platform]: config.actuators[platform] || [] }, platform);
    this.wm = new WorkingMemory(g(config.memory, 'working_memory_capacity', 20));
    this.perceptionCap = g(config.memory, 'perception_capacity', 100);
    this.monologue = new Monologue(defaults.consciousness.max_monologue_length);
    this.emotions = { ...defaults.emotions }; this.drives = { ...defaults.drives };
    this.topics = { ...defaults.topics }; this.lastExplored = {}; this.mode = defaults.initial_thinking_mode;
    this.perceptions = []; this.episodic = 0; this.facts = []; this.users = []; this.messages = [];
    this.llm = {}; this.events = [];
    this.lastProcessed = {};
    this.intervals = g(config.perception, 'process_intervals', defaults.perception_default_intervals);
    const cog = config.cognition || {}, aut = config.autonomy || {}, con = config.consciousness || {};
    this.cogIntervals = { reasoning: g(cog, 'reasoning_interval', 5), emotion: g(cog, 'emotion_interval', 3),
      planning: g(cog, 'planning_interval', 10) };
    this.aut = { base: g(aut, 'thought_interval', defaults.thought_interval), idle: g(aut, 'idle_threshold', defaults.idle_threshold),
      meta: g(aut, 'metacognition_interval', defaults.metacognition_interval),
      min: g(aut, 'min_interest_threshold', defaults.min_interest_threshold) };
    this.con = { minInit: g(con, 'min_initiation_interval', 3600), idleInit: g(con, 'idle_initiation_threshold', 1800),
      enabled: g(con, 'enable_conversation_initiation', true), delay: defaults.consciousness.initiative_startup_delay,
      curiosity: defaults.consciousness.default_state.emotions.curiosity };
    const proc = config.processing || {};
    this.loops = [  // [period (s), handler, next due]
      [g(proc, 'perception_interval', 0.2), () => this.perceptionCycle()],
      [g(proc, 'cognition_interval', 0.5), () => this.cognitionCycle()],
      [g(proc, 'consciousness_interval', 0.3), () => this.consciousnessCycle()],
      [0.5, () => this.autonomyCycle()],
      [TICK, () => this.actuators.tick(this.t)],
    ].map(([p, f]) => ({ period: p, f, next: 0 }));
    this.last = { reasoning: 0, emotion: 0, planning: 0, thought: 0, meta: 0, initiation: 0, conversation: 0, interaction: 0 };
    this.canInitiate = false;
    this.pose = { x: 0, y: 0, th: -Math.PI / 2 }; this.wheels = this.wheelState();
    this.trail = [];
  }

  log(kind, text) { this.events.push({ t: this.t, kind, text }); if (this.events.length > 300) this.events.shift(); }
  askLLM(model, why) { this.llm[model] = (this.llm[model] || 0) + 1; this.lastLLM = { model, why, t: this.t }; }
  addWM(item, imp) { this.episodic += this.wm.add(item, imp, this.t).length; }

  // Advance the simulated clock by dt seconds, running every loop whose period has come round.
  advance(dt) {
    const end = this.t + dt;
    for (;;) {
      const loop = this.loops.reduce((a, b) => (b.next < a.next ? b : a));
      if (loop.next > end) break;
      this.t = loop.next; loop.f(); loop.next += loop.period;
    }
    this.t = end;
    this.updatePose();
  }

  perceptionCycle() {
    this.sensors.advance(this.t);
    const readings = this.sensors.getReadings(this.t);
    for (const [name, d] of Object.entries(readings)) {
      if (d.type === 'camera' && d.description) this.addWM({ type: 'perception', sensor: name, content: `Visual: ${d.description}` }, 0.6);
      else if (d.type === 'microphone' && 'text' in d) this.addWM({ type: 'perception', sensor: name, content: `Audio: ${d.text}` }, 0.7);
    }
    for (const p of perceive(readings, this.intervals, this.lastProcessed, this.t)) {
      this.perceptions.push({ t: this.t, sensor: p.name, type: p.type, data: p.data });
      if (this.perceptions.length > this.perceptionCap) this.perceptions.shift();
      if (p.type === 'camera' && !p.llm) {
        this.addWM({ type: 'perception', sensor: p.name, sensor_type: 'camera', content: `Visual: ${p.data.description}` }, 0.5);
      }
      if (p.llm) this.askLLM(p.llm, `interpret ${p.type} reading`);
    }
  }

  cognitionCycle() {
    const L = this.last, I = this.cogIntervals;
    if (this.t - L.reasoning >= I.reasoning || this.t === 0) { L.reasoning = this.t; this.askLLM('reasoning_engine', 'reason about perceptions'); }
    if (this.t - L.emotion >= I.emotion || this.t === 0) {
      L.emotion = this.t; this.askLLM('emotion_engine', 'update emotions');
      if (this.emotionMode === 'drift') {
        const before = this.emotions;
        this.emotions = driftEmotions(before, this.rng);
        if (significantChange(before, this.emotions)) {
          this.addWM({ type: 'emotion_change', content: { previous: before, current: this.emotions } }, 0.6);
          this.log('emotion', 'significant emotion change stored in working memory');
        }
      }
    }
    if (this.t - L.planning >= I.planning || this.t === 0) { L.planning = this.t; this.askLLM('planning_engine', 'plan goals'); }
  }

  autonomyCycle() {
    const idle = this.t - this.last.interaction;
    if (this.t - this.last.meta >= this.aut.meta) {
      this.last.meta = this.t; this.askLLM('reasoning_engine', 'metacognition'); this.log('autonomy', 'metacognitive reflection requested');
    }
    if (this.t - this.last.thought >= thoughtInterval(this.aut.base, this.aut.idle, idle) || this.last.thought === 0 && this.t > 0) {
      const emo = this.wm.items.find(i => i.type === 'emotion_change');
      this.mode = selectThinkingMode(this.mode, emo ? emo.content.current : null, this.rng);
      const topic = selectTopic(this.topics, this.lastExplored, this.t, this.aut.min, this.rng);
      this.last.thought = this.t;
      this.askLLM('reasoning_engine', `autonomous thought about ${topic} (${this.mode})`);
      this.log('thought', `thinks about ${topic.replaceAll('_', ' ')} in ${this.mode} mode (prompt to reasoning_engine)`);
    }
  }

  consciousnessCycle() {
    if (!this.canInitiate && this.t > this.con.delay) { this.canInitiate = true; this.log('system', 'conversation initiation enabled'); }
    if (!this.con.enabled || !this.canInitiate) return;
    if (this.t - this.last.initiation < this.con.minInit && this.last.initiation > 0) return;
    const idle = this.t - this.last.conversation;
    if (idle < this.con.idleInit) return;
    const hour = Math.floor((START_HOUR * 3600 + this.t) / 3600) % 24;
    if (this.rng() < initiationProbability(this.con.curiosity, idle, hour)) {
      this.last.initiation = this.t;
      this.askLLM('reasoning_engine', 'conversation starter');
      this.send({ id: `initiative_${Math.floor(this.t)}`, content: FALLBACK_STARTER, sender: 'agent', recipient: 'human', initiative: true });
      this.log('initiative', 'agent starts a conversation (LLM unavailable: built-in starter)');
    }
  }

  // ConsciousnessModule.receive_message + _process_incoming_messages, with the LLM reply replaced by the fallback.
  receive(text) {
    const known = this.users.length > 0, msg = { id: `msg_${Math.floor(this.t)}`, sender: 'human', content: text, timestamp: this.t };
    this.last.conversation = this.t; this.last.interaction = this.t;
    this.messages.push({ t: this.t, from: 'human', text });
    const { facts, users } = extractFacts('human', text, known);
    this.facts.push(...facts); this.users.push(...users);
    for (const f of facts) this.log('fact', `stored fact ${f[1]} = ${f[2]}`);
    for (const u of users) this.log('fact', `user name: ${u[1]}`);
    this.addWM({ type: 'incoming_message', content: msg }, 0.8);
    this.monologue.add(`RECEIVED MESSAGE: ${text.slice(0, 50)}...`, clock(this.t));
    this.askLLM('consciousness_engine', 'reply to message');
    this.send({ id: `fallback_${Math.floor(this.t)}`, in_reply_to: msg.id, content: FALLBACK_RESPONSE, sender: 'agent',
      recipient: 'human', fallback: true });
    this.episodic += 1;
  }

  send(response) {
    this.messages.push({ t: this.t, from: 'agent', text: response.content, fallback: !!response.fallback, initiative: !!response.initiative });
    this.addWM({ type: 'outgoing_message', content: response }, 0.7);
    this.episodic += 1;
    this.actuators.communicate(response);
  }

  // ConsciousnessModule._execute_action with the action JSON an LLM would have produced.
  act(action) {
    this.addWM({ type: 'action', content: action }, 0.6);
    this.episodic += 1;
    const ok = this.actuators.executeAction(action.type, action.target, { ...(action.parameters || {}) });
    this.monologue.add(`ACTION: ${action.description || action.type}`, clock(this.t));
    this.log('action', `${action.type} -> ${action.target}: ${ok ? 'sent to actuators' : 'no matching actuator'}`);
    return ok;
  }

  // ---- visualisation only (not in the repo): a differential-drive pose from the two wheel motors' positions ----
  wheelState() {
    const m = this.actuators.byType('motor').map(n => this.actuators.actuators[n]);
    const L = m.find(a => /left/.test(a.name)), R = m.find(a => /right/.test(a.name));
    return L && R ? { L, R, pl: L.state.position, pr: R.state.position } : null;
  }

  updatePose() {
    const w = this.wheels;
    if (!w) return;
    const dl = w.L.state.position - w.pl, dr = w.R.state.position - w.pr;
    w.pl = w.L.state.position; w.pr = w.R.state.position;
    if (!dl && !dr) return;
    const SCALE = 1.0, BASE = 0.5;  // world units per motor position unit; wheel separation
    const p = this.pose, ds = SCALE * (dl + dr) / 2;
    p.th += SCALE * (dr - dl) / BASE;
    p.x = Math.max(-4.6, Math.min(4.6, p.x + ds * Math.cos(p.th)));
    p.y = Math.max(-2.8, Math.min(2.8, p.y + ds * Math.sin(p.th)));
    this.trail.push([p.x, p.y]); if (this.trail.length > 400) this.trail.shift();
  }
}
