// Entry point: lazy page modules, footer mode, subtle background, router.
import { start, lazy, route, app, isStatic, $ } from './core.js';

const PAGES = { 'home.js': ['home'], 'world.js': ['world'], 'architecture.js': ['architecture'], 'mind.js': ['mind'],
  'hardware.js': ['hardware'], 'memory.js': ['memory'], 'dashboard.js': ['dashboard'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));

route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the simulated world runs in your browser; no LLM is called'
  : 'local app: dashboard + live agent (LLM calls go to your local Ollama only)';

// Background: a slow network of nodes (sensors -> mind -> actuators), off under reduced motion.
const bg = $('#bg');
if (bg && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const ctx = bg.getContext('2d');
  const pts = Array.from({ length: 28 }, (_, i) => ({ x: Math.random(), y: Math.random(), p: i * 0.7 }));
  const draw = t => {
    const W = bg.width = innerWidth, H = bg.height = innerHeight;
    ctx.clearRect(0, 0, W, H);
    const P = pts.map(q => [W * (q.x + 0.02 * Math.sin(t / 4000 + q.p)), H * (q.y + 0.02 * Math.cos(t / 5000 + q.p))]);
    ctx.strokeStyle = 'rgba(54,226,255,.18)';
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const d = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]);
      if (d < 170) { ctx.globalAlpha = 1 - d / 170; ctx.beginPath(); ctx.moveTo(...P[i]); ctx.lineTo(...P[j]); ctx.stroke(); }
    }
    ctx.globalAlpha = 1; ctx.fillStyle = 'rgba(176,124,255,.6)';
    P.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); });
    setTimeout(() => requestAnimationFrame(draw), 80);
  };
  requestAnimationFrame(draw);
}
start();
