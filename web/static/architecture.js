// Architecture: data-flow diagram, the processing loops with their intervals (config.yaml), and every module (ast).
import { api, app, esc, route } from './core.js';

const box = (x, y, w, label, sub, cls) => `<g><rect class="box ${cls}" x="${x}" y="${y}" width="${w}" height="54" rx="6"/>
  <text x="${x + w / 2}" y="${y + 23}" text-anchor="middle">${label}</text>
  <text class="sub" x="${x + w / 2}" y="${y + 41}" text-anchor="middle">${sub}</text></g>`;
const edge = (d, cls = '') => `<path class="edge ${cls}" d="${d}"/>`;

function diagram() {
  const W = 190;
  return `<svg class="diagram" viewBox="0 0 990 350" role="img" aria-label="Data flow: sensors to perception to memory to
    cognition to consciousness to actuators; autonomy feeds consciousness; the LLM manager serves the violet modules">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0L10 5L0 10z" fill="#5d6b94"/></marker></defs>
  ${box(10, 30, W, 'SensorManager', 'camera · mic · lidar · …', 's')}
  ${box(270, 30, W, 'PerceptionModule', 'interprets readings', 'm')}
  ${box(530, 30, W, 'MemoryManager', 'working · episodic · SQLite', '')}
  ${box(790, 30, W, 'CognitionModule', 'reason · emotion · plan', 'm')}
  ${box(530, 150, W, 'AutonomyModule', 'thoughts · topics', 'm')}
  ${box(790, 150, W, 'ConsciousnessModule', 'messages · actions', 'm')}
  ${box(790, 270, W, 'ActuatorManager', 'display · speaker · motors', 'a')}
  ${box(530, 270, W, 'human', 'chat: GUI · CLI · web', 's')}
  ${box(10, 150, W, 'LLMManager', 'workers → local Ollama', '')}
  ${edge('M200 57H265')}${edge('M460 57H525')}${edge('M720 57H785')}${edge('M885 84V145')}${edge('M625 84V145')}
  ${edge('M720 177H785')}${edge('M885 204V265')}${edge('M790 297H725')}${edge('M700 270C740 240 790 230 815 209')}
  ${edge('M105 150C110 110 220 100 265 75', 'llm')}${edge('M200 177H525', 'llm')}
  <text class="sub" x="12" y="236">violet boxes send prompts to their own</text>
  <text class="sub" x="12" y="252">LLM role through LLMManager (dashed)</text>
  </svg>`;
}

route('architecture', async () => {
  const [cfg, mods] = await Promise.all([api('/config'), api('/architecture')]);
  const p = cfg.processing, c = cfg.cognition, k = cfg.consciousness, a = cfg.autonomy;
  const roles = Object.entries(cfg.models).map(([r, m]) => `<tr><td class="mono">${esc(r)}</td><td class="mono">${esc(m.model_id)}</td>
    <td>${esc(m.type)}</td><td class="mono small">${esc(m.api_base)}</td></tr>`).join('');
  const groups = {};
  for (const m of mods) (groups[m.path.includes('/') ? m.path.split('/')[0] : 'entry points'] ||= []).push(m);
  const modRows = Object.entries(groups).map(([g, ms]) => `<tr><th colspan="4">${esc(g)}</th></tr>` + ms.map(m => `<tr>
    <td class="mono">${esc(m.path)}</td><td class="mono">${m.lines}</td>
    <td>${m.classes.map(cl => `<b>${esc(cl.name)}</b> <small class="mute">${cl.methods} methods</small>`).join('<br>') || '<span class="mute">–</span>'}</td>
    <td class="small">${esc(m.doc)}${m.imports.length ? `<br><span class="mute">imports ${m.imports.map(esc).join(', ')}</span>` : ''}</td></tr>`).join('')).join('');
  app().innerHTML = `
  <p class="kicker">Architecture · from the code</p><h1>Modules &amp; data flow</h1>
  <p class="lead"><code>EmbodiedAgent</code> (core/agent.py) builds one of each module from config.yaml and runs them in
    parallel threads. Readings flow from the sensors through perception into memory, cognition updates an internal state,
    and the consciousness module talks to the outside world through the actuators, while the autonomy module injects its
    own thoughts.</p>
  <div class="panel">${diagram()}</div>
  <h2>The processing loops</h2>
  <div class="scroll"><table>
    <tr><th>thread</th><th>period</th><th>what one pass does</th></tr>
    <tr><td>perception</td><td class="mono">${p.perception_interval}s</td><td>read every sensor (<code>SensorManager.get_readings</code>);
      camera descriptions and microphone text go straight into working memory; each sensor type is stored and sent to its
      interpreter at most once per <code>perception.process_intervals</code> (camera ${cfg.perception.process_intervals.camera}s,
      microphone ${cfg.perception.process_intervals.microphone}s, lidar ${cfg.perception.process_intervals.lidar}s …)</td></tr>
    <tr><td>cognition</td><td class="mono">${p.cognition_interval}s</td><td>reasoning every ${c.reasoning_interval}s, emotions every
      ${c.emotion_interval}s, planning every ${c.planning_interval}s, each a prompt to its own LLM role</td></tr>
    <tr><td>consciousness</td><td class="mono">${p.consciousness_interval}s</td><td>answer incoming messages, consider starting a
      conversation (at least ${k.min_initiation_interval}s apart, after ${k.idle_initiation_threshold}s without one),
      occasionally turn the plan's first short-term action into actuator commands</td></tr>
    <tr><td>autonomy</td><td class="mono">0.5s</td><td>a thought every ${a.thought_interval}s (more often after ${a.idle_threshold}s
      without human interaction), metacognition every ${a.metacognition_interval}s</td></tr>
    <tr><td>memory worker</td><td class="mono">queue</td><td>writes episodic memories to SQLite in the background</td></tr>
    <tr><td>LLM workers</td><td class="mono">×${cfg.num_workers}</td><td>take queued prompts and call the Ollama API</td></tr>
  </table></div>
  <h2>LLM roles</h2>
  <div class="scroll"><table><tr><th>role</th><th>model</th><th>type</th><th>api_base</th></tr>${roles}</table></div>
  <h2>Every module</h2>
  <p class="small mute">Parsed with Python's <code>ast</code> when this page was built: line counts, classes and internal imports.</p>
  <div class="scroll"><table class="mods">${modRows}</table></div>`;
});
