// Static-site data layer: answers api() GETs from the JSON files exported by scripts/build_static.py (api/info.json,
// api/config.json, api/defaults.json, api/architecture.json, api/schema.json, api/config_yaml.json). The local-only
// routes (/local/...: live agent, memory.db, logs, Ollama) do not exist here; pages show needsServer() instead.
const files = new Map();
function file(rel) {
  if (!files.has(rel)) {
    files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  }
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (body || clean.startsWith('local/')) throw new Error('This needs the local app (python run_web.py).');
  return file(clean);
}
