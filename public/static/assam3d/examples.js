// The "Example: test the model" panel for the Assam page.
//
// It runs entirely on /api/examples, so it works with or without WebGL. When the 3D
// viewer is alive it also flies the camera to the city view; when WebGL is missing the
// panel simply opens over the page and still scores 30 real held-out survey records.
import { GRADES } from '/static/shared/shared.js';

const $ = (id) => document.getElementById(id);
const GRADE_TEXT = { 1: 'Low', 2: 'Medium', 3: 'Near-complete' };
const fmtPct = (v, d = 1) => (v == null ? '–' : (v * 100).toFixed(d) + '%');
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));
const gradeChip = (g) => `<span class="grade" style="background:${(GRADES[g] || {}).hex || '#5a6b7d'}">${g}</span>`;

const ex = { seed: 0, busy: false };

function syncOpen() {
  const open = !$('bpanel').hidden || !$('epanel').hidden;
  $('app').classList.toggle('open', open);
  $('example-btn').setAttribute('aria-expanded', String(!$('epanel').hidden));
}

export function openExamples() {
  // the 3D viewer, when it loaded, tidies the selection and moves the camera
  window.__assam3d?.focusExamples?.();
  $('bpanel').hidden = true;
  $('epanel').hidden = false;
  syncOpen();
  runExamples(ex.seed === 0);
}

export function closeExamples() {
  $('epanel').hidden = true;
  syncOpen();
}

export async function runExamples(fresh) {
  if (fresh || ex.seed === 0) ex.seed = 1 + Math.floor(Math.random() * 999999);
  if (ex.busy) return;
  ex.busy = true;
  const n = Number($('e-n').value);
  $('e-status').textContent = `Running the model on ${n} real records…`;
  try {
    const r = await fetch(`/api/examples?n=${n}&seed=${ex.seed}`);
    if (!r.ok) throw new Error(`server returned ${r.status}`);
    renderExamples(await r.json());
  } catch (e) {
    $('e-status').textContent = `Could not run the examples: ${e.message}`;
  } finally {
    ex.busy = false;
  }
}

export function renderExamples(d) {
  $('e-acc').textContent = fmtPct(d.accuracy);
  $('e-full').textContent = fmtPct(d.test_accuracy, 2);
  $('e-maj').textContent = fmtPct(d.majority_accuracy);
  $('e-status').textContent = `Sample ${d.correct} of ${d.n} correct. The model is ${d.correct === d.n
    ? 'right on every record here'
    : `off on ${d.n - d.correct} here`}. Small samples vary a lot; the full held-out figure is the reliable one.`;

  // confusion matrix: rows are the real grade, columns the predicted grade
  const head = `<tr><th>Real ↓ · Predicted →</th>${[1, 2, 3]
    .map((g) => `<th>${gradeChip(g)} ${GRADE_TEXT[g]}</th>`).join('')}<th>Recall</th></tr>`;
  const body = d.confusion.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    const cells = row.map((c, j) => `<td class="${i === j ? 'diag' : ''}">${c}</td>`).join('');
    return `<tr><th>${gradeChip(i + 1)} ${GRADE_TEXT[i + 1]}</th>${cells}<td>${total ? fmtPct(row[i] / total, 0) : '–'}</td></tr>`;
  }).join('');
  $('e-cm').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;

  const rows = d.rows.map((r) => {
    const inp = r.inputs;
    const walls = inp.walls && inp.walls.length ? inp.walls.map(esc).join(', ') : 'none listed';
    return `<tr>
      <td>${esc(inp.floors)} fl · ${esc(inp.age)} yr · ${esc(walls)}</td>
      <td>${gradeChip(r.actual)}</td>
      <td>${gradeChip(r.predicted)}</td>
      <td>${fmtPct(r.confidence, 0)}</td>
      <td class="${r.correct ? 'ok' : 'miss'}">${r.correct ? '✓ match' : '✗ miss'}</td>
    </tr>`;
  }).join('');
  $('e-rows').innerHTML =
    `<thead><tr><th>Survey inputs</th><th>Real</th><th>Predicted</th><th>Confidence</th><th>Result</th></tr></thead><tbody>${rows}</tbody>`;
  $('e-note').textContent = d.note;
}

// The viewer marks itself once it has a working renderer, so a click that arrives before
// then still opens the panel.
$('example-btn').addEventListener('click', () => {
  if ($('epanel').hidden) openExamples(); else closeExamples();
});
$('eclose').addEventListener('click', closeExamples);
$('e-new').addEventListener('click', () => runExamples(true));
$('e-n').addEventListener('change', () => runExamples(false));
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (!$('epanel').hidden) closeExamples();
});
