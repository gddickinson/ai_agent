// JS port of the framework's non-LLM cognitive logic (core/memory.py, core/autonomy.py, core/cognition.py,
// core/consciousness.py, core/perception.py). Everything an LLM would produce is left out; the simulator shows the
// repo's own no-LLM paths instead (fallback response, fallback emotion drift). Held to the Python results by
// mind_engine.test.mjs (fixture: tests/fixtures/engine_fixture.json).

const get = (o, k, d) => (o && Object.prototype.hasOwnProperty.call(o, k) ? o[k] : d);
const clamp01 = v => Math.max(0, Math.min(1, v));

// ---- MemoryManager working memory -------------------------------------------------
export class WorkingMemory {
  constructor(capacity = 20) { this.capacity = capacity; this.items = []; }
  // add_to_working_memory: insert, sort by importance (stable, high first), drop the least important beyond capacity.
  // Returns the dropped items that MemoryManager would move to episodic memory (importance > 0.4).
  add(item, importance = 0.5, now = 0) {
    if (!('timestamp' in item)) item.timestamp = now;
    item.importance = importance;
    this.items.push(item);
    this.items.sort((a, b) => get(b, 'importance', 0) - get(a, 'importance', 0));
    const spilled = [];
    while (this.items.length > this.capacity) {
      const removed = this.items.pop();
      if (get(removed, 'importance', 0) > 0.4) spilled.push(removed);
    }
    return spilled;
  }
}

// ---- AutonomyModule ------------------------------------------------------------
export const THINKING_MODES = ['analytical', 'creative', 'reflective', 'exploratory', 'emotional', 'predictive'];

export function updateTopics(topics, newTopics) {
  const t = { ...topics };
  const norm = newTopics.map(x => x.toLowerCase().replaceAll(' ', '_'));
  for (const topic of norm) t[topic] = topic in t ? Math.min(1.0, t[topic] + 0.1) : 0.5;
  for (const topic of Object.keys(t)) {
    if (!norm.includes(topic)) {
      t[topic] *= 0.99;
      if (t[topic] < 0.2) delete t[topic];
    }
  }
  return t;
}

export function topicWeights(topics, lastExplored, now, minThreshold) {
  const adj = {};
  for (const [topic, score] of Object.entries(topics)) {
    if (score < minThreshold) continue;
    const timeFactor = Math.min(1.0, (now - get(lastExplored, topic, 0)) / 3600.0);
    adj[topic] = score * (1.0 + timeFactor);
  }
  const total = Object.values(adj).reduce((a, b) => a + b, 0);
  if (total > 0) for (const k of Object.keys(adj)) adj[k] /= total;
  return adj;
}

// _select_topic_of_interest: weighted draw (or uniform over all topics if none qualifies); marks it explored.
export function selectTopic(topics, lastExplored, now, minThreshold, rng) {
  const names = Object.keys(topics);
  if (!names.length) return 'general_cognition';
  const w = topicWeights(topics, lastExplored, now, minThreshold);
  let topic;
  if (!Object.keys(w).length) topic = names[Math.floor(rng() * names.length)];
  else {
    let r = rng(), acc = 0;
    const keys = Object.keys(w);
    topic = keys[keys.length - 1];
    for (const k of keys) { acc += w[k]; if (r < acc) { topic = k; break; } }
  }
  lastExplored[topic] = now;
  return topic;
}

export function thoughtInterval(base, idleThreshold, idleTime) {
  return idleTime > idleThreshold ? Math.max(5.0, base * (0.8 - idleTime / 1000.0)) : base;
}

// _select_thinking_mode: `emotions` from the newest 'emotion_change' item in working memory, or null.
export function selectThinkingMode(current, emotions, rng) {
  if (emotions) {
    const e = k => get(emotions, k, 0);
    if (e('curiosity') > 0.6) return 'exploratory';
    if (e('confusion') > 0.6) return 'analytical';
    if (e('joy') > 0.6) return 'creative';
    if (e('sadness') > 0.4 || e('fear') > 0.4) return 'reflective';
    return current;
  }
  return rng() < 0.3 ? THINKING_MODES[Math.floor(rng() * THINKING_MODES.length)] : current;
}

// ---- CognitionModule -----------------------------------------------------------
export const EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'curiosity', 'confusion'];

function pyFloat(v) {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'string' && v.trim() !== '' && !isNaN(Number(v.trim()))) return Number(v.trim());
  return null;
}

// _parse_emotion_output: the JSON between the first '{' and last '}', each emotion clamped to [0, 1]. Without JSON
// the Python code drifts the current emotions by up to +-0.1 (the path the simulator uses with no LLM).
export function parseEmotions(text, current, rng) {
  const s = text.indexOf('{'), e = text.lastIndexOf('}');
  if (s >= 0 && e > s) {
    let obj;
    try { obj = JSON.parse(text.slice(s, e + 1)); } catch { return { ...current }; }
    const out = {};
    for (const k of EMOTIONS) { const v = pyFloat(get(obj, k, 0.0)); out[k] = v === null ? 0.0 : clamp01(v); }
    return out;
  }
  return driftEmotions(current, rng);
}

export function driftEmotions(current, rng) {
  const out = {};
  for (const [k, v] of Object.entries(current)) out[k] = clamp01(v + (rng() - 0.5) * 0.2);
  return out;
}

export const significantChange = (before, after) =>
  Object.entries(after).some(([k, v]) => Math.abs(v - get(before, k, 0.0)) > 0.3);

// ---- ConsciousnessModule ---------------------------------------------------------
export function initiationProbability(curiosity, idleTime, hour) {
  const p = Math.min(0.8, curiosity + idleTime / (3600.0 * 3));
  return 9 <= hour && hour <= 22 ? p * 1.5 : p * 0.5;
}

const COMMON_NAMES = ['George', 'John', 'Mary', 'David', 'Sarah', 'Michael', 'Emma', 'James', 'Alice', 'Bob', 'Peter',
  'Susan', 'Tom', 'Linda', 'Richard', 'Emily'];
const capitalize = s => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();

// receive_message + _extract_facts: regex facts and the user's name. Returns {facts: [[entity, attr, value]], users}.
export function extractFacts(sender, content, knownUser = false) {
  const facts = [], users = [], ent = `user:${sender}`, low = content.toLowerCase();
  if (low.includes('my age is')) {
    const m = low.match(/my age is (\d+)/);
    if (m) facts.push([ent, 'age', m[1]]);
  }
  for (const p of [/I am (\d+)(?:\s+years old)?/i, /I'm (\d+)(?:\s+years old)?/i, /my age is (\d+)/i, /my age: (\d+)/i]) {
    const m = content.match(p);
    if (m) { facts.push([ent, 'age', m[1]]); break; }
  }
  const pats = [[/I(?:'m| am) (?:from|in) ([A-Za-z\s,]+)/, 'location'], [/I (?:like|love|enjoy) ([A-Za-z\s,]+)/, 'likes'],
    [/I (?:dislike|hate|don't like) ([A-Za-z\s,]+)/, 'dislikes']];
  for (const [re, attr] of pats) {
    const m = content.match(re);
    if (m) facts.push([ent, attr, m[1].trim()]);
  }
  const nm = low.match(/my name is ([\p{L}\p{N}_]+)/u);
  if (nm) users.push([sender, capitalize(nm[1])]);
  else if (!knownUser) {
    const hit = COMMON_NAMES.find(n => low.includes(n.toLowerCase()));
    if (hit) users.push([sender, hit]);
  }
  return { facts, users };
}

export const FALLBACK_RESPONSE = "I'm sorry, I'm having trouble processing your message right now. My language " +
  'processing system is experiencing a delay. Please try again or rephrase your query.';

export class Monologue {
  constructor(max = 50) { this.max = max; this.lines = []; }
  add(thought, clock) {
    const t = String(thought).trim();
    if (!t) return;
    this.lines.push(`[${clock}] ${t}`);
    while (this.lines.length > this.max) this.lines.shift();
  }
}

// ---- PerceptionModule.process ------------------------------------------------------
// Which readings are processed now (per-type process_intervals) and which interpreter LLM each would be sent to.
export function perceive(readings, intervals, lastProcessed, now) {
  const out = [];
  for (const [name, data] of Object.entries(readings)) {
    const type = get(data, 'type', 'unknown');
    if (!(type in intervals) || now - get(lastProcessed, type, -Infinity) < intervals[type]) continue;
    let llm = null;
    if (type === 'camera') llm = data.simulated && 'description' in data ? null : 'vision_interpreter';
    else if (type === 'microphone') llm = get(data, 'text', '') ? 'audio_interpreter' : null;
    else llm = 'general_interpreter';
    out.push({ name, type, data, llm });
    lastProcessed[type] = now;
  }
  return out;
}
