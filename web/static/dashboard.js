// Dashboard (local app only): run the real EmbodiedAgent, chat with it through the local Ollama server, watch its
// state, and read the display/speaker output logs. The static site shows what the dashboard does instead.
import { api, app, esc, route, isStatic, needsServer, onLeave, $ } from './core.js';

const bars = obj => Object.entries(obj || {}).map(([k, v]) => `<div class="bar-row"><span>${esc(k.replaceAll('_', ' '))}</span>
  <i style="--v:${(Math.min(1, +v || 0) * 100).toFixed(1)}%"></i><b>${(+v || 0).toFixed(2)}</b></div>`).join('');

function staticPage() {
  app().innerHTML = `<p class="kicker">Dashboard · local app</p><h1>Dashboard</h1>
    ${needsServer('The dashboard')}
    <p class="lead">Run locally, this page starts the real agent from config.yaml (all sensors forced to simulation, so no
      webcam is opened), lets you chat with it and send actions, and shows its monologue, emotions, topics, working memory,
      memory statistics and LLM queue. It also checks which configured Ollama models your local server has and tails the
      display and speaker output files. Every LLM call goes to the <code>api_base</code> in config.yaml (localhost).</p>
    <p><a class="btn" href="#/world">Try the simulated world instead</a></p>`;
}

route('dashboard', async () => {
  if (isStatic()) return staticPage();
  app().innerHTML = `<p class="kicker">Dashboard · local app</p><h1>Dashboard</h1>
    <div class="two">
      <div class="panel"><h3>Ollama</h3><div id="ollama" class="small">checking…</div></div>
      <div class="panel"><h3>Agent</h3><div class="row"><div><label for="plat">Platform</label>
        <select id="plat"><option>mac</option><option>robot</option></select></div>
        <div style="flex:0"><button id="start">Start</button></div><div style="flex:0"><button id="stop" class="alt">Stop</button></div></div>
        <p id="ag-status" class="small mute"></p>
        <p class="small mute">Uses <code>memory/memory.db</code> like main.py: on start the agent deletes conversation facts and
          low-importance episodes older than 30 days. Sensors are forced to simulation.</p></div>
    </div>
    <div class="two">
      <div class="panel"><h3>Chat</h3><div id="chat" class="chat"></div>
        <form id="chatf" class="row"><div><label for="msg">Message</label><input id="msg" autocomplete="off"></div>
        <div style="flex:0"><button>Send</button></div></form>
        <h3>Action</h3><div class="row"><div><label for="atype">Type</label><select id="atype"><option>speak</option><option>display</option><option>move</option></select></div>
        <div><label for="atext">Text / direction</label><input id="atext" value="Hello"></div><div style="flex:0"><button id="act" class="alt">Execute</button></div></div></div>
      <div class="panel"><h3>Internal monologue</h3><pre id="mono" class="mono-log"></pre><h3>Thinking</h3><div id="think" class="small"></div></div>
    </div>
    <div class="three">
      <div class="panel"><h3>Emotions</h3><div id="emo"></div><h3>Drives</h3><div id="drv"></div></div>
      <div class="panel"><h3>Topics of interest</h3><div id="top"></div></div>
      <div class="panel"><h3>Working memory</h3><ol id="wm" class="wm"></ol><h3>Memory &amp; LLM</h3><div id="stats" class="small mono"></div></div>
    </div>
    <h2>Output logs</h2><div id="logs" class="two"></div>`;

  api('/local/ollama', { fresh: true }).then(o => {
    $('#ollama').innerHTML = o.servers.map(s => `<p><span class="status-dot ${s.up ? 'up' : ''}"></span><code>${esc(s.api_base)}</code>
      ${s.up ? `${s.installed.length} models installed` : `not reachable (${esc(s.error)})`}</p>`).join('') +
      `<table class="tight"><tr><th>role</th><th>model</th><th>installed</th></tr>${o.roles.map(r => `<tr><td class="mono">${esc(r.role)}</td>
      <td class="mono">${esc(r.model_id)}</td><td class="${r.installed ? 'ok' : 'err'}">${r.installed ? 'yes' : 'no'}</td></tr>`).join('')}</table>`;
  }).catch(e => { $('#ollama').textContent = e.message; });

  const logs = async () => {
    const l = await api('/local/logs?lines=40', { fresh: true });
    $('#logs').innerHTML = l.files.map(f => `<div class="panel"><h3>${esc(f.name)}</h3><p class="small mute">${esc(f.path)}${f.exists ? '' : ' (not created yet)'}</p>
      <pre class="mono-log">${esc(f.lines.join('\n'))}</pre></div>`).join('') || '<p class="mute">No output files configured.</p>';
  };

  const render = s => {
    $('#ag-status').textContent = s.running ? `running on ${s.platform}: sensors ${s.sensors.map(x => x[0]).join(', ')}; actuators ${s.actuators.map(x => x[0]).join(', ')}` : 'stopped';
    $('#chat').innerHTML = (s.messages || []).map(m => `<p class="msg ${m.from}"><b>${m.from}</b> ${esc(m.text)}${m.fallback ? ' <small class="mute">(fallback)</small>' : ''}</p>`).join('');
    if (!s.running) return;
    $('#mono').textContent = s.monologue.join('\n');
    $('#think').innerHTML = `mode <b class="mono">${esc(s.thinking_mode || '–')}</b>${(s.questions || []).length ? '<br>questions: ' + s.questions.map(esc).join(' · ') : ''}`;
    $('#emo').innerHTML = bars(s.emotions); $('#drv').innerHTML = bars(s.drives); $('#top').innerHTML = bars(s.topics);
    $('#wm').innerHTML = s.working_memory.map(w => `<li><span class="tag">${(+w.importance).toFixed(1)}</span> <b>${esc(w.type)}</b> <span class="mute">${esc(w.content)}</span></li>`).join('');
    $('#stats').textContent = JSON.stringify({ memory: s.memory_stats, llm: s.llm }, null, 1);
  };
  const poll = async () => { try { render(await api('/local/agent', { fresh: true })); } catch (e) { $('#ag-status').textContent = e.message; } };
  const post = async (path, body) => { try { await api(path, { body }); } catch (e) { $('#ag-status').textContent = e.message; } poll(); };

  $('#start').onclick = () => { $('#ag-status').textContent = 'starting…'; post('/local/agent/start', { platform: $('#plat').value }); };
  $('#stop').onclick = () => post('/local/agent/stop', {});
  $('#chatf').onsubmit = e => { e.preventDefault(); const t = $('#msg').value.trim(); if (t) { $('#msg').value = ''; post('/local/agent/message', { text: t }); } };
  $('#act').onclick = () => {
    const t = $('#atype').value, v = $('#atext').value;
    const params = t === 'move' ? { direction: v || 'forward', speed: 0.5 } : { text: v };
    post('/local/agent/action', { type: t, target: 'all', parameters: params, description: `${t} from the dashboard` });
  };
  poll(); logs();
  const id = setInterval(() => { poll(); logs(); }, 2000);
  onLeave(() => clearInterval(id));
});
