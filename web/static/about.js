// About: what the project is, how to run the desktop and web apps, credits.
import { api, app, esc, route } from './core.js';

route('about', async () => {
  const i = await api('/info');
  app().innerHTML = `
  <p class="kicker">About</p><h1>Embodied AI Agent</h1>
  <p class="lead">${esc(i.description)}</p>
  <p>The architecture models perception, memory, cognition (reasoning, emotion, planning), a "consciousness" module that
    handles communication and an autonomy module for self-directed thought. Each uses a local LLM through Ollama
    (${i.model_ids.map(m => `<code>${esc(m)}</code>`).join(', ')} in config.yaml). The hardware layer adapts to the
    platform and falls back to simulated sensors and actuators.</p>
  <h2>Run the agent</h2>
  <pre>pip install -r requirements.txt     # pyyaml, requests (+ opencv-python for a real webcam)
ollama pull llama3                  # and llama3.2-vision for camera images
python main.py --hardware mac       # or --hardware robot; -c config.yaml; --debug
python enhanced_chat.py             # terminal chat
python -m gui.app                   # Tkinter GUI</pre>
  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py                   # http://127.0.0.1:8132</pre>
  <p>The local web app adds the <a href="#/dashboard">Dashboard</a>: start the real agent (sensors forced to simulation), chat
    with it through your own Ollama server, browse <code>memory/memory.db</code> and the display/speaker output logs. LLM
    calls go only to the <code>api_base</code> in config.yaml (localhost).</p>
  <h2>This site</h2>
  <p>The static site never calls an LLM. Its <a href="#/world">simulated world</a> runs JavaScript ports of the framework's
    non-LLM logic (motor model, action routing, simulated sensors, perception intervals, working memory, emotion parsing and
    drift, topic interest, thinking modes, conversation-initiation odds, fact extraction). Node tests hold each port to the
    values the Python code computes (<code>scripts/make_engine_fixture.py</code>). The architecture pages are generated
    from the code itself: modules are parsed with <code>ast</code>, the schema is created by the real
    <code>MemoryManager</code>, defaults come from the real modules.</p>
  <h2>Credits</h2>
  <p>By George Dickinson. MIT licence. From the README: inspired by various cognitive architectures and research in
    artificial general intelligence; built using Ollama and LLM models for cognitive processing; built with AI assistance
    from Claude (Anthropic).</p>`;
});
