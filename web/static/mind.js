// Cognition & autonomy: the rules the framework applies without an LLM, each with a live calculator that uses the
// node-tested JS port (mind_engine.js).
import { api, app, esc, route, $ } from './core.js';
import { updateTopics, topicWeights, thoughtInterval, initiationProbability, extractFacts, parseEmotions,
  THINKING_MODES } from './mind_engine.js';
import { rng } from './core_rng.js';

const bars = obj => Object.entries(obj).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<div class="bar-row"><span>${esc(k.replaceAll('_', ' '))}</span>
  <i style="--v:${Math.min(100, v * 100).toFixed(1)}%"></i><b>${v.toFixed(3)}</b></div>`).join('');

function intervalChart(base, idleThr) {
  const W = 520, H = 160, X = t => 40 + t / 3000 * (W - 50), Y = v => H - 24 - v / base * (H - 40);
  let d = '';
  for (let t = 0; t <= 3000; t += 10) d += `${t ? 'L' : 'M'}${X(t).toFixed(1)} ${Y(thoughtInterval(base, idleThr, t)).toFixed(1)}`;
  return `<svg class="diagram" viewBox="0 0 ${W} ${H}" role="img" aria-label="Thought interval against idle time">
    <path d="${d}" stroke="var(--accent)" stroke-width="2" fill="none"/>
    <line x1="40" x2="${W - 10}" y1="${H - 24}" y2="${H - 24}" stroke="#5d6b94"/><line x1="40" x2="40" y1="10" y2="${H - 24}" stroke="#5d6b94"/>
    <text class="sub" x="40" y="${H - 6}">0</text><text class="sub" x="${W - 60}" y="${H - 6}">3000 s idle</text>
    <text class="sub" x="44" y="${Y(base) - 4}">${base} s</text><text class="sub" x="${X(1500)}" y="${Y(5) - 6}">floor 5 s</text></svg>`;
}

route('mind', async () => {
  const [d, cfg] = await Promise.all([api('/defaults'), api('/config')]);
  const a = cfg.autonomy;
  app().innerHTML = `
  <p class="kicker">Architecture · core/cognition.py, core/autonomy.py, core/consciousness.py</p><h1>Cognition &amp; autonomy</h1>
  <p class="lead">The LLMs write the thoughts, plans and replies; around them the modules keep numeric state and apply
    fixed rules. These are those rules, with calculators that run the same JavaScript as the simulated world.</p>
  <div class="two">
    <div class="panel"><h3>Emotions at start</h3>${bars(d.emotions)}</div>
    <div class="panel"><h3>Drives at start</h3>${bars(d.drives)}</div>
  </div>
  <p>Every ${cfg.cognition.emotion_interval}s the emotion engine is asked for a new JSON emotion state. Values are clamped to
    [0, 1] and missing emotions become 0. If the reply has no JSON, each emotion drifts by a random amount up to ±0.1; if
    the LLM fails, nothing changes. A change of more than 0.3 in any emotion is stored in working memory (importance 0.6).</p>
  <div class="panel"><label for="emo-in">Try an emotion-engine reply</label>
    <input id="emo-in" value='Feeling: {"joy": 0.9, "curiosity": "0.7", "fear": -1, "anger": "high"}'>
    <div id="emo-out"></div></div>

  <h2>Thinking modes</h2>
  <p>${THINKING_MODES.map(m => `<span class="tag ${m === d.initial_thinking_mode ? 'a' : ''}">${m}</span>`).join('')}
    Before each autonomous thought the mode follows the newest emotion change in working memory: curiosity &gt; 0.6 →
    exploratory, else confusion &gt; 0.6 → analytical, else joy &gt; 0.6 → creative, else sadness or fear &gt; 0.4 →
    reflective. With no emotion change on record there is a 30% chance of switching to a random mode.</p>

  <h2>Topics of interest</h2>
  <p>The agent starts with ${Object.keys(d.topics).length} topics. When a thought names topics, each named topic gains 0.1
    (max 1) or enters at 0.5; every other topic is multiplied by 0.99 and dropped below 0.2. To choose what to think
    about, topics at or above ${a.min_interest_threshold} are weighted by score × (1 + hours since last explored, capped at 1).</p>
  <div class="two">
    <div class="panel"><label for="top-in">Topics named by a thought (comma separated)</label>
      <input id="top-in" value="Robotics, learning">
      <button id="top-go">Apply update</button><button id="top-reset" class="alt">Reset</button>
      <div id="top-out"></div></div>
    <div class="panel"><h3>Selection probability (none explored yet)</h3><div id="w-out"></div></div>
  </div>

  <h2>When it thinks</h2>
  <p>A thought every ${a.thought_interval}s; after ${a.idle_threshold}s without human interaction the interval becomes
    max(5, ${a.thought_interval} × (0.8 − idle/1000)), so a quiet agent thinks more often.</p>
  <div class="panel">${intervalChart(a.thought_interval, a.idle_threshold)}</div>

  <h2>When it starts a conversation</h2>
  <p>Not in the first ${d.consciousness.initiative_startup_delay}s, at least ${cfg.consciousness.min_initiation_interval}s after the
    last one, and only after ${cfg.consciousness.idle_initiation_threshold}s without a human message. Then, on each check,
    the chance is min(0.8, curiosity + idle hours / 3), times 1.5 from 09:00 to 22:59 and 0.5 at night. The consciousness
    module reads curiosity from an <code>internal_state</code> item in working memory; no module writes one, so its built-in
    default (${d.consciousness.default_state.emotions.curiosity}) is what is used.</p>
  <div class="panel row">
    <div><label for="ic">Curiosity</label><input id="ic" type="number" min="0" max="1" step="0.1" value="${d.consciousness.default_state.emotions.curiosity}"></div>
    <div><label for="ii">Idle minutes</label><input id="ii" type="number" min="0" step="5" value="15"></div>
    <div><label for="ih">Hour</label><input id="ih" type="number" min="0" max="23" value="14"></div>
    <div><p id="ip" class="eq"></p></div></div>

  <h2>What it learns from a message</h2>
  <p>Without any LLM, <code>receive_message</code> stores regex facts about the sender (age, location, likes, dislikes) and a
    name ("my name is …", or one of 16 common first names found anywhere in the text).</p>
  <div class="panel"><label for="fx">Message</label><input id="fx" value="Hi, my name is Ada. I'm from Leeds and I like chess, I hate rain">
    <div id="fx-out"></div></div>`;

  const emo = () => { $('#emo-out').innerHTML = bars(parseEmotions($('#emo-in').value, d.emotions, rng(1))); };
  let topics = { ...d.topics };
  const top = () => {
    $('#top-out').innerHTML = bars(topics);
    $('#w-out').innerHTML = bars(topicWeights(topics, {}, 3600, a.min_interest_threshold));
  };
  const ip = () => {
    const p = initiationProbability(+$('#ic').value, +$('#ii').value * 60, +$('#ih').value);
    $('#ip').textContent = `p = ${p.toFixed(3)} per check`;
  };
  const fx = () => {
    const { facts, users } = extractFacts('human', $('#fx').value);
    $('#fx-out').innerHTML = (facts.map(f => `<span class="tag a">${esc(f[1])}: ${esc(f[2])}</span>`).join('') +
      users.map(u => `<span class="tag b">name: ${esc(u[1])}</span>`).join('')) || '<span class="mute small">nothing extracted</span>';
  };
  $('#emo-in').oninput = emo; $('#fx').oninput = fx;
  ['#ic', '#ii', '#ih'].forEach(s => { $(s).oninput = ip; });
  $('#top-go').onclick = () => { topics = updateTopics(topics, $('#top-in').value.split(',').map(s => s.trim()).filter(Boolean)); top(); };
  $('#top-reset').onclick = () => { topics = { ...d.topics }; top(); };
  emo(); top(); ip(); fx();
});
