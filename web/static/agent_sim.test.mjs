// node web/static/agent_sim.test.mjs: the simulated agent wires the ported pieces together as core/agent.py does.
import assert from 'node:assert/strict';
import { AgentSim, clock } from './agent_sim.js';
import { FALLBACK_RESPONSE } from './mind_engine.js';
import { rng } from './core_rng.js';

const config = {
  platforms: ['robot'], memory: { working_memory_capacity: 20, perception_capacity: 100 },
  sensors: { robot: [{ name: 'camera', type: 'camera', frame_interval: 1, simulate: true },
    { name: 'microphone', type: 'microphone', capture_interval: 2, simulate: true },
    { name: 'battery', type: 'battery', read_interval: 30, simulate: true, simulate_min: 0, simulate_max: 100, simulate_unit: '%' }] },
  actuators: { robot: [{ name: 'screen', type: 'display' }, { name: 'speaker', type: 'speaker' },
    { name: 'left_wheel', type: 'motor' }, { name: 'right_wheel', type: 'motor' }] },
  perception: { process_intervals: { camera: 5, microphone: 3, battery: 30 } },
  cognition: { reasoning_interval: 5, emotion_interval: 3, planning_interval: 10 },
  consciousness: { min_initiation_interval: 1800, idle_initiation_threshold: 900, enable_conversation_initiation: true },
  autonomy: { thought_interval: 20, idle_threshold: 120, metacognition_interval: 600, min_interest_threshold: 0.4 },
  processing: { perception_interval: 0.2, cognition_interval: 0.5, consciousness_interval: 0.3 },
};
const defaults = { emotions: { joy: 0, curiosity: 0.5 }, drives: { social: 0.5 }, topics: { learning: 0.9, creativity: 0.5 },
  initial_thinking_mode: 'reflective', thought_interval: 20, idle_threshold: 120, metacognition_interval: 600,
  min_interest_threshold: 0.4, perception_default_intervals: {},
  consciousness: { max_monologue_length: 50, initiative_startup_delay: 300, default_state: { emotions: { curiosity: 0.8 } } } };

const sim = new AgentSim({ config, defaults, platform: 'robot', rng: rng(5) });
sim.advance(60);
assert.ok(sim.perceptions.some(p => p.type === 'camera') && sim.perceptions.some(p => p.type === 'battery'));
assert.ok(sim.wm.items.some(i => i.content && String(i.content).startsWith('Visual: ')));
assert.ok(sim.llm.emotion_engine >= 20 && sim.llm.reasoning_engine > 0, 'LLM prompts counted, not sent');
assert.ok(sim.events.some(e => e.kind === 'thought'));

sim.receive('Hi, my name is Ada and I am 40');
assert.deepEqual(sim.users, [['human', 'Ada']]); assert.deepEqual(sim.facts, [['user:human', 'age', '40']]);
assert.equal(sim.messages.at(-1).text, FALLBACK_RESPONSE);
sim.advance(0.3);
assert.equal(sim.actuators.actuators.screen.output.at(-1).text, FALLBACK_RESPONSE);
assert.equal(sim.actuators.actuators.speaker.output.at(-1).text, FALLBACK_RESPONSE);

const x0 = sim.pose.x;
assert.equal(sim.act({ type: 'move', target: 'all', parameters: { direction: 'forward', speed: 1 } }), true);
sim.advance(1);
assert.ok(sim.actuators.actuators.left_wheel.state.position > 0.8 && sim.pose.x !== x0, 'wheels turn, robot moves');
assert.equal(sim.act({ type: 'move', target: 'forward', parameters: {} }), false, 'unknown target: no motor moves');

// idle long enough: the agent starts a conversation with the built-in starter
sim.advance(2000);
assert.ok(sim.messages.some(m => m.initiative), 'conversation initiated after the idle threshold');
assert.equal(clock(3725), '11:02:05');
console.log('agent sim: wiring checks pass');
