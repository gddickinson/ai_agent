// Simulated world: the agent's non-LLM machinery (sensors, perception, working memory, emotions, autonomy, actuators)
// running in the browser on a simulated clock. See agent_sim.js for what replaces each LLM call.
import { api, app, esc, route, animate, rng, $ } from './core.js';
import { AgentSim } from './agent_sim.js';
import * as V from './world_view.js';

const SPEEDS = [1, 10, 60];
let st = { platform: 'robot', speed: 10, paused: false, emotionMode: 'drift', seed: 7 };

route('world', async (args, query) => {
  const [config, defaults] = await Promise.all([api('/config'), api('/defaults')]);
  if (query.platform && config.platforms.includes(query.platform)) st.platform = query.platform;
  let sim;
  const reset = () => { sim = new AgentSim({ config, defaults, platform: st.platform, rng: rng(st.seed), emotionMode: st.emotionMode }); };
  reset();

  app().innerHTML = `
  <div class="simhead"><div><p class="kicker">Simulated world · no LLM, no hardware</p><h1>The agent, running</h1></div>
    <div class="bar">
      <span class="seg" role="group" aria-label="Platform">${config.platforms.map(p => `<button data-plat="${p}">${p}</button>`).join('')}</span>
      <span class="seg" role="group" aria-label="Simulation speed">${SPEEDS.map(s => `<button data-speed="${s}">${s}×</button>`).join('')}</span>
      <button class="alt" id="pause">Pause</button><button class="alt" id="reset">Reset</button>
    </div></div>
  <p class="small mute">The ${esc(st.platform)} platform's sensors and actuators come from config.yaml and run the JS ports of the
    repo's simulations. Every prompt the Python agent would send to Ollama is counted, not sent; the agent answers with the
    repo's own no-LLM fallbacks. The room, the robot's pose and its trail are drawn by this page from the two wheel motors'
    positions: the repo models each motor as a 1-D position, not a robot in a room.</p>
  <div class="world">
    <div>
      <div class="stage"><canvas id="room" width="1040" height="620" aria-label="Simulated room with the agent"></canvas></div>
      <div class="panel ctlpanel">
        <h3>Act (you play the action LLM)</h3>
        <div class="bar" id="drive">
          <button data-a="fwd">Forward</button><button data-a="back">Back</button>
          <button data-a="left">Turn left</button><button data-a="right">Turn right</button><button data-a="stop" class="alt">Stop</button>
        </div>
        <div class="row">
          <div><label for="say">Text to speak or display</label><input id="say" value="Hello, I am online."></div>
          <div style="flex:0"><button id="speak">Speak</button></div><div style="flex:0"><button id="show" class="alt">Display</button></div>
        </div>
        <h3>Talk to the agent</h3>
        <form id="chatf" class="row"><div><label for="msg">Message (try “My name is Ada, I'm from Leeds and I like chess”)</label>
          <input id="msg" autocomplete="off"></div><div style="flex:0"><button>Send</button></div></form>
        <div id="chat"></div>
        <label for="emo">Emotion engine stand-in</label>
        <select id="emo"><option value="drift">LLM replies without JSON: the repo's random drift (±0.1)</option>
          <option value="frozen">Ollama down: emotions stay as they are</option></select>
      </div>
    </div>
    <div class="side-panels">
      <div class="panel"><h3>Sensors → perception</h3><div id="p-sens"></div></div>
      <div class="panel"><h3>Motors</h3><div id="p-mot"></div></div>
      <div class="panel"><h3>Emotions</h3><div id="p-emo"></div></div>
      <div class="panel"><h3>Topics of interest</h3><div id="p-top"></div></div>
    </div>
  </div>
  <div class="three">
    <div class="panel"><h3>Working memory</h3><div id="p-wm"></div></div>
    <div class="panel"><h3>Internal monologue</h3><div id="p-mono"></div><h3>LLM prompts (not sent)</h3><div id="p-llm"></div></div>
    <div class="panel"><h3>Events</h3><div id="p-ev"></div></div>
  </div>`;

  const sync = () => {
    document.querySelectorAll('[data-plat]').forEach(b => b.classList.toggle('on', b.dataset.plat === st.platform));
    document.querySelectorAll('[data-speed]').forEach(b => b.classList.toggle('on', +b.dataset.speed === st.speed));
    $('#pause').textContent = st.paused ? 'Run' : 'Pause';
    $('#emo').value = st.emotionMode;
  };
  sync();
  document.querySelectorAll('[data-plat]').forEach(b => b.onclick = () => { st.platform = b.dataset.plat; reset(); sync(); panels(); });
  document.querySelectorAll('[data-speed]').forEach(b => b.onclick = () => { st.speed = +b.dataset.speed; sync(); });
  $('#pause').onclick = () => { st.paused = !st.paused; sync(); };
  $('#reset').onclick = () => { st.seed += 1; reset(); panels(); };
  $('#emo').onchange = e => { st.emotionMode = sim.emotionMode = e.target.value; };

  const move = (target, direction, speed = 0.5) => sim.act({ type: 'move', target, parameters: { direction, speed },
    description: `move ${target} ${direction}` });
  const drive = {
    fwd: () => move('all', 'forward'), back: () => move('all', 'backward'),
    left: () => { move('left_wheel', 'backward', 0.2); move('right_wheel', 'forward', 0.2); },
    right: () => { move('left_wheel', 'forward', 0.2); move('right_wheel', 'backward', 0.2); },
    stop: () => move('all', 'stop', 0),
  };
  document.querySelectorAll('#drive [data-a]').forEach(b => b.onclick = () => { drive[b.dataset.a](); panels(); });
  $('#speak').onclick = () => { sim.act({ type: 'speak', target: 'all', parameters: { text: $('#say').value }, description: 'speak' }); panels(); };
  $('#show').onclick = () => { sim.act({ type: 'display', target: 'all', parameters: { text: $('#say').value }, description: 'display' }); panels(); };
  $('#chatf').onsubmit = e => {
    e.preventDefault();
    const t = $('#msg').value.trim();
    if (t) { sim.receive(t); $('#msg').value = ''; panels(); }
  };

  const canvas = $('#room'), ctx = canvas.getContext('2d');
  function panels() {
    const set = (id, html) => { const el = $(id); if (el) el.innerHTML = html; };
    set('#p-sens', V.sensorsPanel(sim)); set('#p-mot', V.motorsPanel(sim)); set('#p-emo', V.emotionsPanel(sim));
    set('#p-top', V.topicsPanel(sim)); set('#p-wm', V.memoryPanel(sim)); set('#p-mono', V.monologuePanel(sim));
    set('#p-llm', V.llmPanel(sim)); set('#p-ev', V.eventsPanel(sim)); set('#chat', V.chatPanel(sim));
  }
  let last = 0, lastPanel = 0;
  animate(now => {
    const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
    last = now;
    if (!st.paused && dt) sim.advance(dt * st.speed);
    V.drawWorld(ctx, sim, canvas.width, canvas.height);
    if (now - lastPanel > 250) { lastPanel = now; panels(); }
  });
  panels();
});
