// node web/static/mind_engine.test.mjs: the JS cognitive logic reproduces core/memory.py, core/autonomy.py,
// core/cognition.py and core/consciousness.py (fixture: tests/fixtures/engine_fixture.json).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { WorkingMemory, updateTopics, topicWeights, thoughtInterval, selectThinkingMode, selectTopic, parseEmotions,
  driftEmotions, initiationProbability, extractFacts, perceive, Monologue } from './mind_engine.js';
import { rng } from './core_rng.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/engine_fixture.json', import.meta.url)));
const close = (a, b, what) => assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${what}: js ${a} vs py ${b}`);
const closeDict = (a, b, what) => {
  assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), `${what} keys`);
  for (const k of Object.keys(b)) close(a[k], b[k], `${what}.${k}`);
};
let n = 0;

// working memory
const wmf = fx.working_memory, wm = new WorkingMemory(wmf.capacity), spilled = [];
wmf.adds.forEach(([name, imp], i) => {
  spilled.push(...wm.add({ type: name, timestamp: 0 }, imp));
  assert.deepEqual(wm.items.map(x => x.type), wmf.snapshots[i], `working memory after add ${i}`); n++;
});
assert.deepEqual(spilled.map(x => x.type), wmf.spilled, 'items moved to episodic memory');

// autonomy
const au = fx.autonomy;
const DEFAULTS = au.default_topics;
let t = DEFAULTS;
au.updates.forEach((u, i) => { t = updateTopics(t, u); closeDict(t, au.after[i], `topics after update ${i}`); n++; });
let d = { ...au.decay_start };
au.decay.forEach((want, i) => { d = updateTopics(d, ['y']); closeDict(d, want, `decay ${i}`); n++; });
for (const c of au.weights) {
  closeDict(topicWeights(c.topics || DEFAULTS, c.explored, c.now, au.min_interest_threshold), c.weights, `weights @${c.now}`); n++;
}
for (const [idle, want] of au.intervals) { close(thoughtInterval(au.thought_interval, au.idle_threshold, idle), want, `interval ${idle}`); n++; }
for (const [emo, mode] of au.modes) { assert.equal(selectThinkingMode('predictive', emo, rng(1)), mode, JSON.stringify(emo)); n++; }

// cognition
for (const [text, want] of fx.cognition.parsed) { closeDict(parseEmotions(text, {}, rng(1)), want, text); n++; }
const cur = { joy: 0.5, curiosity: 0.99, fear: 0.0 }, r = rng(7);
for (let i = 0; i < 200; i++) for (const [k, v] of Object.entries(driftEmotions(cur, r))) {
  assert.ok(v >= 0 && v <= 1 && Math.abs(v - cur[k]) <= 0.1 + 1e-12, 'drift stays within +-0.1 and [0, 1]');
}

// consciousness
for (const [c, idle, hour, p] of fx.consciousness.initiation) { close(initiationProbability(c, idle, hour), p, `p(${c},${idle},${hour})`); n++; }
for (const m of fx.consciousness.messages) {
  const got = extractFacts('human', m.text);
  assert.deepEqual(got.facts, m.facts, `facts: ${m.text}`); assert.deepEqual(got.users, m.users, `name: ${m.text}`); n++;
}

// semantics checked without Python (random draws use a different generator)
const seen = {}, last = {}, rr = rng(3);
for (let i = 0; i < 2000; i++) { const tp = selectTopic({ a: 0.9, b: 0.45, c: 0.1 }, {}, 0, 0.4, rr); seen[tp] = (seen[tp] || 0) + 1; }
assert.ok(!seen.c && seen.a > seen.b * 1.6, 'topics below threshold are never chosen; higher scores more often');
assert.equal(selectTopic({ only: 0.9 }, last, 42, 0.4, rr), 'only'); assert.equal(last.only, 42);
const lp = {}, ints = { camera: 5, microphone: 3, lidar: 2 };
const rd = { cam: { type: 'camera', simulated: true, description: 'x' }, mic: { type: 'microphone', text: 'hi' },
  bat: { type: 'battery', value: 1 } };
assert.deepEqual(perceive(rd, ints, lp, 0).map(p => [p.name, p.llm]), [['cam', null], ['mic', 'audio_interpreter']]);
assert.deepEqual(perceive(rd, ints, lp, 2.9), []);
assert.deepEqual(perceive(rd, ints, lp, 3).map(p => p.name), ['mic']);
const mono = new Monologue(3); ['a', ' ', 'b', 'c', 'd'].forEach(x => mono.add(x, '00:00:00'));
assert.deepEqual(mono.lines, ['[00:00:00] b', '[00:00:00] c', '[00:00:00] d']);
console.log(`mind engine: ${n} checks match the Python code`);
