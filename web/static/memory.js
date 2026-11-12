// Memory: the memory systems and their rules, the SQLite schema (created by the real MemoryManager), and on the local
// app a read-only browser for memory/memory.db.
import { api, app, esc, route, isStatic, needsServer, $ } from './core.js';

const IMPORTANCE = [['incoming message', 0.8, 'consciousness'], ['microphone text', 0.7, 'agent perception loop'],
  ['outgoing reply', 0.7, 'consciousness'], ['plan', 0.7, 'cognition'], ['camera description', 0.6, 'agent perception loop'],
  ['emotion change > 0.3', 0.6, 'cognition'], ['action', 0.6, 'consciousness'], ['snapshot request', 0.6, 'consciousness'],
  ['simulated camera perception', 0.5, 'perception'], ['reasoning result', 'from the LLM (default 0.5)', 'cognition'],
  ['autonomous thought', 'from the LLM, only if > 0.6', 'autonomy']];

route('memory', async () => {
  const [schema, cfg] = await Promise.all([api('/schema'), api('/config')]);
  const tables = schema.map(t => `<div class="card"><b class="mono">${esc(t.name)}</b>
    <span class="mono small">${t.columns.map(c => `${esc(c.name)} <span class="mute">${esc(c.type)}${c.pk ? ' pk' : ''}</span>`).join('<br>')}</span>
    ${t.indexes.length ? `<small>index: ${t.indexes.map(esc).join(', ')}</small>` : ''}</div>`).join('');
  app().innerHTML = `
  <p class="kicker">Architecture · core/memory.py</p><h1>Memory</h1>
  <p class="lead"><code>MemoryManager</code> keeps two in-process lists and a SQLite database
    (<code>${esc(cfg.memory.memory_dir)}/memory.db</code>), written from a background worker thread.</p>
  <div class="two">
    <div class="panel"><h3>Working memory · ${cfg.memory.working_memory_capacity} items</h3>
      <p class="small">Every item carries an importance. After each insert the list is sorted (most important first) and the
      least important items beyond capacity are dropped; dropped items above 0.4 are written to episodic memory. This is
      what the LLM prompts see as “working memory”.</p>
      <table class="tight"><tr><th>item</th><th>importance</th><th>added by</th></tr>
      ${IMPORTANCE.map(([a, b, c]) => `<tr><td>${a}</td><td class="mono">${b}</td><td class="small mute">${c}</td></tr>`).join('')}</table></div>
    <div class="panel"><h3>Perception memory · ${cfg.memory.perception_capacity} items</h3>
      <p class="small">Interpreted perceptions from the vision, audio and general interpreters, oldest dropped first; raw sensor
      readings also go to the <code>perceptions</code> table.</p>
      <h3>Long-term</h3>
      <p class="small"><b>Episodic</b>: timestamped episodes (communication, thoughts, actions, monologue excerpts) with an
      importance and an access count. <b>Semantic</b>: concepts with a confidence. <b>Facts</b>: entity / attribute / value triples, such as the
      user's age or likes from regex extraction and conversation summaries. <b>Conversations</b> and <b>users</b>: the chat
      transcript and known names.</p></div>
  </div>
  <h2>Database schema</h2>
  <p class="small mute">Created by <code>MemoryManager</code> in a temporary directory when this page was built.</p>
  <div class="grid">${tables}</div>
  <h2>Decay and consolidation</h2>
  <p class="small">Each read of an episode raises its <code>access_count</code> and resets <code>last_accessed</code>. At start-up
    the agent deletes episodes not accessed for 30 days with importance below 0.7 (conversation summaries are kept), and
    conversation facts older than 30 days. Consolidation copies episodes with importance above 0.7 that were read more
    than three times into semantic memory. Databases from before schema version 1 are migrated once: the old
    <code>episodic_memories</code> table is merged into <code>episodic_memory</code> and every existing episode counts as
    accessed at migration time.</p>
  <h2>Your memory database</h2><div id="local"></div>`;

  if (isStatic()) { $('#local').innerHTML = needsServer('Browsing memory/memory.db'); return; }
  const ov = await api('/local/memory', { fresh: true });
  if (!ov.exists) { $('#local').innerHTML = `<p class="mute">No database yet at <code>${esc(ov.path)}</code>: run the agent once.</p>`; return; }
  $('#local').innerHTML = `<p class="small mute">${esc(ov.path)} (read-only)</p>
    <div class="bar">${Object.entries(ov.counts).map(([t, n]) => `<button class="alt" data-t="${t}">${t} · ${n ?? '–'}</button>`).join('')}</div>
    <div id="rows" class="scroll"></div>`;
  document.querySelectorAll('[data-t]').forEach(b => b.onclick = async () => {
    const r = await api(`/local/memory/${b.dataset.t}?limit=50`, { fresh: true });
    $('#rows').innerHTML = r.rows.length ? `<table class="tight"><tr>${r.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr>
      ${r.rows.map(row => `<tr>${row.map(v => `<td class="small">${esc(String(v ?? '').slice(0, 300))}</td>`).join('')}</tr>`).join('')}</table>`
      : '<p class="mute">Empty.</p>';
  });
});
