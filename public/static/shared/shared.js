// Shared pieces for the 3D pages: grade colours, the building form, and a smooth prediction client.
export const GRADES = {
  1: { label: 'Low damage', short: 'Grade 1 · low', hex: '#2f855a', rgb: [0.184, 0.522, 0.353] },
  2: { label: 'Medium damage', short: 'Grade 2 · medium', hex: '#d69e2e', rgb: [0.839, 0.620, 0.180] },
  3: { label: 'Near-complete destruction', short: 'Grade 3 · near-complete', hex: '#c53030', rgb: [0.773, 0.188, 0.188] },
};

// Fields a user can change in the 3D pages. Everything else keeps the dataset default.
const NUMERIC = [
  { name: 'count_floors_pre_eq', label: 'Floors before earthquake', min: 1, max: 9, step: 1 },
  { name: 'age', label: 'Building age (years)', min: 0, max: 200, step: 1 },
  { name: 'area_percentage', label: 'Plinth area (% of plot)', min: 1, max: 100, step: 1 },
  { name: 'height_percentage', label: 'Height (survey units)', min: 2, max: 32, step: 1 },
  { name: 'count_families', label: 'Families in building', min: 0, max: 9, step: 1 },
];
const CHOICES = ['land_surface_condition', 'ground_floor_type', 'roof_type', 'foundation_type'];

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export async function loadOptions() {
  const r = await fetch('/api/options');
  if (!r.ok) throw new Error('options ' + r.status);
  return r.json();
}

// Build the controls inside `root`. onChange(payload) is called with a full 38-field payload.
export function buildControls(root, options, onChange) {
  const by = Object.fromEntries(options.fields.map((f) => [f.name, f]));
  const defaults = Object.fromEntries(options.fields.map((f) => [f.name, f.default]));
  const materials = options.fields.filter((f) => f.kind === 'binary' && f.name.startsWith('has_superstructure_') && !f.name.endsWith('_other'));

  const numHTML = NUMERIC.map((n) => {
    const f = by[n.name];
    return `<div class="ctl"><label><span>${esc(n.label)}</span><output data-out="${n.name}">${f.default}</output></label>
      <input type="range" name="${n.name}" min="${n.min}" max="${n.max}" step="${n.step}" value="${f.default}"></div>`;
  }).join('');
  const choiceHTML = CHOICES.map((name) => {
    const f = by[name];
    return `<div class="ctl sel"><label><span>${esc(f.label)}</span></label>
      <select name="${name}">${f.values.map((v) => `<option value="${esc(v)}"${String(v) === String(f.default) ? ' selected' : ''}>${esc(v)}</option>`).join('')}</select></div>`;
  }).join('');
  const matHTML = materials.map((f) => `<label class="chip"><input type="checkbox" name="${f.name}" value="1"${Number(f.default) === 1 ? ' checked' : ''}><span>${esc(f.label.replace('Has superstructure ', ''))}</span></label>`).join('');

  root.innerHTML = `
    <fieldset class="grp"><legend>Building</legend>${numHTML}</fieldset>
    <fieldset class="grp"><legend>Site and construction</legend>${choiceHTML}</fieldset>
    <fieldset class="grp"><legend>Structural materials (tick all that apply)</legend><div class="chips">${matHTML}</div></fieldset>
    <p class="fine">Location IDs are fixed to dataset defaults, and so are the less common fields. Survey location IDs cannot be derived from coordinates.</p>`;

  const read = () => {
    const p = { ...defaults };
    root.querySelectorAll('input[type=range]').forEach((el) => {
      p[el.name] = Number(el.value);
      const out = root.querySelector(`output[data-out="${el.name}"]`);
      if (out) out.textContent = el.value;
    });
    root.querySelectorAll('select').forEach((el) => { p[el.name] = el.value; });
    root.querySelectorAll('input[type=checkbox]').forEach((el) => { p[el.name] = el.checked ? 1 : 0; });
    return p;
  };
  root.addEventListener('input', () => onChange(read()));
  root.addEventListener('change', () => onChange(read()));
  return { read, defaults, materials: materials.map((f) => f.name) };
}

// Prediction client: debounced, cached, and drops stale responses.
export function createPredictor({ onResult, onLoading, onError, delay = 260 }) {
  const cache = new Map();
  let timer = null, seq = 0;

  async function run(payload) {
    const my = ++seq;
    const key = JSON.stringify(payload);
    if (cache.has(key)) { onResult(cache.get(key)); return; }
    onLoading && onLoading(true);
    try {
      const r = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: key,
      });
      const body = await r.json().catch(() => null);
      if (!r.ok) {
        const d = body && body.detail;
        throw new Error(typeof d === 'string' ? d : 'Prediction failed (' + r.status + ')');
      }
      cache.set(key, body);
      if (cache.size > 800) cache.delete(cache.keys().next().value);
      if (my === seq) onResult(body);
    } catch (e) {
      if (my === seq && onError) onError(e.message);
    } finally {
      if (my === seq && onLoading) onLoading(false);
    }
  }
  return {
    schedule(payload) { clearTimeout(timer); timer = setTimeout(() => run(payload), delay); },
    now(payload) { clearTimeout(timer); run(payload); },
  };
}

// Animated probability bars inside `el`.
export function renderProbBars(el, probs) {
  el.innerHTML = [1, 2, 3].map((k) => `
    <div class="pbar"><span>Grade ${k}</span>
      <div class="track"><div class="fill" data-k="${k}" style="background:${GRADES[k].hex};width:0%"></div></div>
      <b>${(probs[k] * 100).toFixed(1)}%</b></div>`).join('');
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.querySelectorAll('.fill').forEach((f) => { f.style.width = (probs[f.dataset.k] * 100).toFixed(1) + '%'; });
  }));
}

export const SHARED_CSS = `
  .ctl { margin: 8px 0 10px; }
  .ctl label { display: flex; justify-content: space-between; font-size: 13px; color: #3b4856; }
  .ctl output { font-variant-numeric: tabular-nums; font-weight: 600; color: #14213d; }
  .ctl input[type=range] { width: 100%; accent-color: #b7791f; }
  .ctl.sel select { width: 100%; padding: 6px 8px; border-radius: 7px; border: 1px solid #cfd7e1; font: 14px system-ui, sans-serif; margin-top: 4px; }
  .grp { border: 1px solid #dfe5ec; border-radius: 10px; padding: 8px 12px 10px; margin: 10px 0; }
  .grp legend { font: 700 13px Georgia, serif; color: #14213d; padding: 0 6px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; color: #3b4856; background: #f3f6fb; border: 1px solid #dfe6ef; border-radius: 999px; padding: 3px 9px; cursor: pointer; user-select: none; }
  .chip input { accent-color: #b7791f; margin: 0; }
  .chip:has(input:checked) { background: #fbf3e4; border-color: #e7c98c; color: #6b4610; }
  .fine { font-size: 12px; color: #6b7785; margin: 6px 0 0; }
  .pbar { display: grid; grid-template-columns: 62px 1fr 52px; gap: 8px; align-items: center; font-size: 13px; margin: 6px 0; }
  .pbar .track { height: 10px; background: #eef2f6; border-radius: 6px; overflow: hidden; }
  .pbar .fill { height: 100%; border-radius: 6px; transition: width .6s cubic-bezier(.2,.8,.2,1); }
  .pbar b { text-align: right; font-variant-numeric: tabular-nums; color: #14213d; }
  .badge { border-radius: 12px; padding: 12px 14px; color: #fff; transition: background .4s ease; }
  .badge .k { font-size: 12px; opacity: .92; letter-spacing: .3px; text-transform: uppercase; }
  .badge .v { font: 700 20px Georgia, serif; line-height: 1.2; margin-top: 2px; }
  .spin { display: inline-block; width: 12px; height: 12px; border: 2px solid #c9d3df; border-top-color: #14213d; border-radius: 50%; animation: spin .8s linear infinite; vertical-align: -2px; margin-left: 6px; opacity: 0; transition: opacity .2s; }
  .spin.on { opacity: 1; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .topnav { position: fixed; top: 0; left: 0; right: 0; height: 56px; z-index: 50; display: flex; align-items: center; gap: 4px; padding: 0 16px; background: linear-gradient(90deg, #14213d, #1f3358); box-shadow: 0 2px 12px rgba(20,33,61,.18); font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
  .topnav .brand { color: #fff; font: 700 17px Georgia, "Times New Roman", serif; letter-spacing: .4px; margin-right: 14px; text-decoration: none; }
  .topnav .brand span { color: #d69e2e; margin: 0 3px; }
  .topnav a { color: rgba(255,255,255,.8); text-decoration: none; font-size: 14px; padding: 8px 12px; border-radius: 7px; transition: background .2s, color .2s; }
  .topnav a:hover { background: rgba(255,255,255,.1); color: #fff; }
  .topnav a.on { color: #fff; background: rgba(183,121,31,.28); box-shadow: inset 0 -2px 0 #d69e2e; }
  @media (max-width: 560px) { .topnav a:not(.brand):not(.on) { display: none; } .topnav .brand { margin-right: auto; } }
`;
