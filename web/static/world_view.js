// Drawing for the simulated world: the room canvas and the state panels. Pure rendering of an AgentSim.
import { esc } from './core.js';
import { clock } from './agent_sim.js';

const C = { floor: '#070b16', grid: 'rgba(54,226,255,.07)', wall: '#1d2846', body: '#36e2ff', wheel: '#b07cff',
  trail: 'rgba(54,226,255,.25)', text: '#e8f1ff', mute: '#94a3c6', warm: '#ffb347' };

// World units: the room is 10 x 6, centred on 0.
export function drawWorld(ctx, sim, W, H) {
  const sx = W / 10.4, ox = W / 2, oy = H / 2, P = (x, y) => [ox + x * sx, oy + y * sx];
  ctx.fillStyle = C.floor; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  for (let x = -5; x <= 5; x++) { const [a] = P(x, 0); ctx.beginPath(); ctx.moveTo(a, 0); ctx.lineTo(a, H); ctx.stroke(); }
  for (let y = -3; y <= 3; y++) { const [, b] = P(0, y); ctx.beginPath(); ctx.moveTo(0, b); ctx.lineTo(W, b); ctx.stroke(); }
  ctx.strokeStyle = C.wall; ctx.lineWidth = 3;
  const [x0, y0] = P(-5, -3.1); ctx.strokeRect(x0, y0, 10 * sx, 6.2 * sx);

  // trail and agent body
  if (sim.trail.length > 1) {
    ctx.strokeStyle = C.trail; ctx.lineWidth = 2; ctx.beginPath();
    sim.trail.forEach(([x, y], i) => { const [a, b] = P(x, y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    ctx.stroke();
  }
  const { x, y, th } = sim.pose, [cx, cy] = P(x, y), r = 0.32 * sx;
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(th);
  ctx.shadowColor = C.body; ctx.shadowBlur = 16;
  ctx.fillStyle = 'rgba(54,226,255,.16)'; ctx.strokeStyle = C.body; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.shadowBlur = 0;
  if (sim.wheels) {
    ctx.fillStyle = C.wheel;
    for (const [side, m] of [[-1, sim.wheels.L], [1, sim.wheels.R]]) {
      ctx.globalAlpha = m.state.speed ? 1 : 0.55;
      ctx.fillRect(-r * 0.45, side * r * 0.95 - r * 0.12, r * 0.9, r * 0.24);
    }
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = C.warm; ctx.beginPath(); ctx.moveTo(r * 0.3, 0); ctx.lineTo(r * 0.95, 0); ctx.stroke();
  ctx.restore();

  // what the actuators last produced: speech bubble and the display screen
  const speech = lastOf(sim, 'speaker'), disp = lastOf(sim, 'display');
  ctx.font = `${Math.max(11, sx * 0.2)}px ui-monospace, Menlo, monospace`; ctx.textBaseline = 'top';
  if (speech && sim.t - speech.time < 8) bubble(ctx, cx + r, cy - r * 2.2, speech.text, W);
  const [dx, dy] = P(-4.85, -2.95);
  ctx.fillStyle = 'rgba(176,124,255,.12)'; ctx.strokeStyle = C.wheel; ctx.lineWidth = 1.5;
  const dw = Math.min(W * 0.42, 4.2 * sx), dh = 0.9 * sx;
  ctx.fillRect(dx, dy, dw, dh); ctx.strokeRect(dx, dy, dw, dh);
  ctx.fillStyle = C.mute; ctx.fillText(disp ? 'display' : 'display (no output yet)', dx + 6, dy + 4);
  if (disp) { ctx.fillStyle = C.text; wrap(ctx, disp.text, dx + 6, dy + 4 + sx * 0.25, dw - 12, 2); }
  ctx.fillStyle = C.mute; ctx.textAlign = 'right';
  ctx.fillText(`${clock(sim.t)}  ·  ${sim.platform}`, W - 8, H - sx * 0.3); ctx.textAlign = 'left';
}

function lastOf(sim, type) {
  for (const n of sim.actuators.byType(type)) {
    const o = sim.actuators.actuators[n].output;
    if (o.length) return o[o.length - 1];
  }
  return null;
}

function wrap(ctx, text, x, y, w, maxLines) {
  const words = String(text).split(/\s+/), lh = parseInt(ctx.font, 10) * 1.3;
  let line = '', n = 0;
  for (const word of words) {
    const t = line ? line + ' ' + word : word;
    if (ctx.measureText(t).width > w && line) {
      if (++n >= maxLines) { ctx.fillText(line + ' …', x, y); return; }
      ctx.fillText(line, x, y); y += lh; line = word;
    } else line = t;
  }
  if (line) ctx.fillText(line, x, y);
}

function bubble(ctx, x, y, text, W) {
  const w = Math.min(260, W * 0.45), lh = parseInt(ctx.font, 10) * 1.3, h = lh * 3 + 10;
  const bx = Math.min(x, W - w - 6);
  ctx.fillStyle = 'rgba(255,179,71,.14)'; ctx.strokeStyle = C.warm; ctx.lineWidth = 1.5;
  ctx.fillRect(bx, y - h, w, h); ctx.strokeRect(bx, y - h, w, h);
  ctx.fillStyle = C.text; wrap(ctx, text, bx + 6, y - h + 5, w - 12, 3);
}

// ---- panels ---------------------------------------------------------------------------
const bar = (label, v, cls = '') => `<div class="bar-row ${cls}"><span>${esc(label.replaceAll('_', ' '))}</span>
  <i style="--v:${(v * 100).toFixed(1)}%"></i><b>${v.toFixed(2)}</b></div>`;

export function sensorsPanel(sim) {
  const rows = Object.entries(sim.sensors.sensors).map(([n, s]) => {
    const last = [...sim.perceptions].reverse().find(p => p.sensor === n);
    const val = !last ? '<span class="mute">waiting</span>' : s.type === 'camera' ? esc(last.data.description)
      : s.type === 'microphone' ? `“${esc(last.data.text || '(silence)')}”`
      : `${last.data.value.toFixed(1)} ${esc(last.data.unit)}`;
    return `<tr><td>${esc(n)}<small class="mute"> ${esc(s.type)}${s.realRequested ? ' · real requested, simulated' : ''}</small></td>
      <td class="mono">${s.interval}s</td><td>${val}</td></tr>`;
  }).join('');
  return `<table class="tight"><tr><th>sensor</th><th>every</th><th>last perception</th></tr>${rows}</table>`;
}

export const emotionsPanel = sim => Object.entries(sim.emotions).map(([k, v]) => bar(k, v)).join('');

export function topicsPanel(sim) {
  const t = Object.entries(sim.topics).sort((a, b) => b[1] - a[1]);
  return `<p class="small">Thinking mode: <b class="mono">${esc(sim.mode)}</b></p>` +
    t.map(([k, v]) => bar(k, v, v < sim.aut.min ? 'dim' : '')).join('');
}

export function memoryPanel(sim) {
  const items = sim.wm.items.map(i => `<li><span class="tag ${i.importance >= 0.7 ? 'a' : i.importance >= 0.6 ? 'b' : 'c'}">${i.importance.toFixed(1)}</span>
    <b>${esc(i.type)}</b> <span class="mute">${esc(summary(i))}</span></li>`).join('');
  return `<p class="small mute">${sim.wm.items.length}/${sim.wm.capacity} items, sorted by importance · ${sim.episodic} sent to episodic memory · ${sim.perceptions.length} perceptions stored</p><ol class="wm">${items}</ol>`;
}

function summary(i) {
  const c = i.content;
  if (typeof c === 'string') return c.slice(0, 90);
  if (i.type === 'emotion_change') return 'emotion state changed by more than 0.3';
  if (c && c.content) return String(c.content).slice(0, 90);
  if (c && c.type) return `${c.type} -> ${c.target}`;
  return '';
}

export const monologuePanel = sim => sim.monologue.lines.length
  ? `<pre class="mono-log">${esc(sim.monologue.lines.slice(-12).join('\n'))}</pre>` : '<p class="mute small">Empty: send a message or an action.</p>';

export function llmPanel(sim) {
  const rows = Object.entries(sim.llm).sort((a, b) => b[1] - a[1])
    .map(([m, n]) => `<tr><td class="mono">${esc(m)}</td><td class="mono">${n}</td></tr>`).join('');
  return `<table class="tight"><tr><th>model role</th><th>prompts not sent</th></tr>${rows}</table>`;
}

export function eventsPanel(sim) {
  return `<ul class="events">${sim.events.slice(-14).reverse().map(e => `<li><span class="mono mute">${clock(e.t)}</span>
    <span class="tag">${esc(e.kind)}</span> ${esc(e.text)}</li>`).join('')}</ul>`;
}

export function chatPanel(sim) {
  const facts = sim.facts.map(f => `<span class="tag a">${esc(f[1])}: ${esc(f[2])}</span>`).join('') +
    sim.users.map(u => `<span class="tag b">name: ${esc(u[1])}</span>`).join('');
  return `<div class="chat">${sim.messages.slice(-8).map(m => `<p class="msg ${m.from}"><b>${m.from}</b> ${esc(m.text)}
    ${m.fallback ? '<small class="mute">(fallback: no LLM)</small>' : ''}${m.initiative ? '<small class="mute">(agent-initiated)</small>' : ''}</p>`).join('')}</div>
    ${facts ? `<p class="small">Remembered: ${facts}</p>` : ''}`;
}

export function motorsPanel(sim) {
  const m = sim.actuators.byType('motor');
  if (!m.length) return `<p class="small mute">No motors on the ${esc(sim.platform)} platform (config.yaml): move actions return false.</p>`;
  return `<table class="tight"><tr><th>motor</th><th>speed</th><th>position</th><th>target</th></tr>${m.map(n => {
    const s = sim.actuators.actuators[n].state;
    return `<tr><td>${esc(n)}</td><td class="mono">${s.speed.toFixed(2)}</td><td class="mono">${s.position.toFixed(2)}</td><td class="mono">${s.target ?? '–'}</td></tr>`;
  }).join('')}</table>`;
}
