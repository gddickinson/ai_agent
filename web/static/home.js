// Home page: hero, counts from the code, entry cards.
import { api, app, esc, route } from './core.js';

route('home', async () => {
  const i = await api('/info');
  app().innerHTML = `
  <section class="hero"><p class="kicker">Perception · memory · cognition · consciousness · autonomy</p>
    <h1 class="glow">Embodied AI Agent</h1>
    <p class="lead">${esc(i.description)} Each cognitive function is a module with its own local LLM role (Ollama), wired
      to sensors and actuators through a hardware layer that simulates whatever a platform does not have.</p>
    <a class="btn" href="#/world">Run the simulated world</a><a class="btn alt" href="#/architecture">How it is built</a></section>
  <div class="tiles">
    <div class="tile"><b>${i.modules}</b><span>Python modules (${i.lines.toLocaleString()} lines)</span></div>
    <div class="tile"><b>${i.models}</b><span>LLM roles in config.yaml</span></div>
    <div class="tile"><b>${i.platforms.length}</b><span>hardware platforms (${i.platforms.map(esc).join(', ')})</span></div>
    <div class="tile"><b>${i.sensors.robot} + ${i.actuators.robot}</b><span>robot sensors + actuators</span></div>
    <div class="tile"><b>${i.emotions}</b><span>emotions in the internal state</span></div>
    <div class="tile"><b>${i.tables}</b><span>SQLite memory tables</span></div>
  </div>
  <h2>Explore</h2>
  <div class="grid">
    <a class="card" href="#/world"><b>Simulated world</b><small>runs in your browser</small><span>Sensors, perception, working
      memory, emotions, topic interest and actuators, ported from the Python code and tested against it. Drive the robot,
      talk to the agent, watch what it would ask its LLMs.</span></a>
    <a class="card" href="#/architecture"><b>Modules &amp; data flow</b><small>${i.classes} classes</small><span>The
      processing loops, which module calls which LLM role, and every module of the framework.</span></a>
    <a class="card" href="#/mind"><b>Cognition &amp; autonomy</b><small>the non-LLM rules</small><span>Emotions and drives,
      thinking modes, how topics of interest grow and fade, when the agent starts a conversation.</span></a>
    <a class="card" href="#/hardware"><b>Sensors &amp; actuators</b><small>${i.platforms.length} platforms</small><span>What each
      platform has, what is real hardware and what is simulated, and how actions are routed.</span></a>
    <a class="card" href="#/memory"><b>Memory</b><small>${i.tables} tables</small><span>Working memory, episodic and semantic
      memory, facts and conversations: the rules and the database schema.</span></a>
    <a class="card" href="#/dashboard"><b>Dashboard</b><small>local app</small><span>Start the real agent, chat with it through
      your local Ollama, browse its memory and output logs.</span></a>
  </div>`;
});
