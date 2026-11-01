// "Example: test the model" panel for the world 3D page.
//
// The globe predicts a damage grade for a building profile you describe. This panel
// checks that model the honest way: it scores real survey records the model never saw
// during training and compares each prediction with the grade the survey recorded.
// All of the numbers come from /api/examples; nothing is hard-coded here.

const $ = (id) => document.getElementById(id);

const GRADE_TEXT = { 1: 'Low', 2: 'Medium', 3: 'Near-complete' };
const GRADE_HEX = { 1: '#2f855a', 2: '#d69e2e', 3: '#c53030' };

const state = { seed: 1 + Math.floor(Math.random() * 999999), busy: false };

const esc = (v) => String(v).replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const pct = (v, digits = 1) => (v == null ? '–' : `${(v * 100).toFixed(digits)}%`);

const chip = (g) => `<span class="grade" style="background:${GRADE_HEX[g] || '#5a6b7d'}">${g}</span>`;

const app = $('app');

function sheetOpen() { return app.dataset.sheetOpen === '1'; }

function syncOpen() {
  const examplesOpen = !$('epanel').hidden;
  app.classList.toggle('open', sheetOpen() || examplesOpen);
  $('example-btn').setAttribute('aria-expanded', String(examplesOpen));
}

function openExamples() {
  window.dispatchEvent(new Event('globe:examples-open'));   // the point sheet closes
  $('side').classList.add('examples');                      // and steps out of the column
  $('epanel').hidden = false;
  syncOpen();
  runExamples(true);
}

function closeExamples() {
  $('epanel').hidden = true;
  $('side').classList.remove('examples');
  syncOpen();
}

async function runExamples(fresh) {
  if (state.busy) return;
  if (fresh) state.seed = 1 + Math.floor(Math.random() * 999999);
  state.busy = true;
  const n = Number($('e-n').value);
  $('e-status').textContent = `Running the model on ${n} real records…`;
  try {
    const res = await fetch(`/api/examples?n=${n}&seed=${state.seed}`);
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    render(await res.json());
  } catch (err) {
    $('e-status').textContent = `Could not run the examples: ${err.message}`;
  } finally {
    state.busy = false;
  }
}

function render(d) {
  $('e-acc').textContent = pct(d.accuracy);
  $('e-full').textContent = pct(d.test_accuracy, 2);
  $('e-maj').textContent = pct(d.majority_accuracy);
  $('e-status').textContent =
    `Sample ${d.correct} of ${d.n} correct. The model is ${d.correct === d.n
      ? 'right on every record here'
      : `off on ${d.n - d.correct} here`}. Small samples vary a lot; the full held-out figure is the reliable one.`;

  // confusion matrix: rows are the grade the survey recorded, columns the prediction
  const head = `<tr><th>Real ↓ · Predicted →</th>${[1, 2, 3]
    .map((g) => `<th>${chip(g)} ${GRADE_TEXT[g]}</th>`).join('')}<th>Recall</th></tr>`;
  const body = d.confusion.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    const cells = row.map((c, j) => `<td class="${i === j ? 'diag' : ''}">${c}</td>`).join('');
    return `<tr><th>${chip(i + 1)} ${GRADE_TEXT[i + 1]}</th>${cells}<td>${total ? pct(row[i] / total, 0) : '–'}</td></tr>`;
  }).join('');
  $('e-cm').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;

  const rows = d.rows.map((r) => {
    const inp = r.inputs;
    const walls = inp.walls && inp.walls.length ? inp.walls.map(esc).join(', ') : 'none listed';
    return `<tr>
      <td>${esc(inp.floors)} fl · ${esc(inp.age)} yr · ${walls}</td>
      <td>${chip(r.actual)}</td>
      <td>${chip(r.predicted)}</td>
      <td>${pct(r.confidence, 0)}</td>
      <td class="${r.correct ? 'ok' : 'miss'}">${r.correct ? '✓ match' : '✗ miss'}</td>
    </tr>`;
  }).join('');
  $('e-rows').innerHTML =
    `<thead><tr><th>Survey inputs</th><th>Real</th><th>Predicted</th><th>Confidence</th><th>Result</th></tr></thead>`
    + `<tbody>${rows}</tbody>`;
  $('e-note').textContent = d.note;
}

$('example-btn').addEventListener('click', () => {
  if ($('epanel').hidden) openExamples(); else closeExamples();
});
$('eclose').addEventListener('click', closeExamples);
$('e-new').addEventListener('click', () => runExamples(true));
$('e-n').addEventListener('change', () => runExamples(false));
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('epanel').hidden) closeExamples();
});

// the globe script announces when a point is picked; the example panel steps aside
window.addEventListener('globe:sheet', () => {
  if (sheetOpen() && !$('epanel').hidden) $('epanel').hidden = true;
  syncOpen();
});
