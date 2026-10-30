// Assam in 3D: Overture Maps building footprints, streamed as 0.1-degree tiles.
// Click a building to predict its damage grade. The model is the same one the lab uses.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GRADES, SHARED_CSS, loadOptions, buildControls, createPredictor, renderProbBars } from '/static/shared/shared.js';

document.getElementById('shared-css').textContent = SHARED_CSS;

const BASE = '/static/assam3d/tiles/';
const REF_LON = 91.7, REF_LAT = 26.2;                  // projection centre (metres from here)
const KX = 111320 * Math.cos(REF_LAT * Math.PI / 180), KY = 110574;
const DETAIL_MAX = 160000;                              // most building outlines in memory at once (nearest tiles first)
const DETAIL_MIN = 40000;                               // the floor, on slow devices
let detailBudget = DETAIL_MAX;                          // adapts to the frame rate (see the loop)
const DETAIL_MAX_DIST = 60000;                          // above this camera distance, overview only
const CONCURRENCY = 3;
const PENDING_RGB = [0.48, 0.52, 0.60];                 // neutral colour until a grade is known
const PENDING_HEX = '#7a8699';
const HOME = { x: 92.9, lat: 26.1 };

const project = (lon, lat) => [(lon - REF_LON) * KX, -(lat - REF_LAT) * KY];   // [x, z]
const unproject = (x, z) => [REF_LON + x / KX, REF_LAT - z / KY];              // [lon, lat]
const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n).toLocaleString('en-US');

// ---------------------------------------------------------------- renderer and scene
const container = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#dfe8ef');
scene.fog = new THREE.Fog('#dfe8ef', 1500000, 4000000);

const camera = new THREE.PerspectiveCamera(40, 1, 5, 5e6);
scene.add(new THREE.HemisphereLight('#ffffff', '#9aa6b4', 1.6));
const sun = new THREE.DirectionalLight('#ffffff', 1.2);
sun.position.set(-1, 2, 1);
scene.add(sun);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 150;
controls.maxDistance = 1200000;
controls.maxPolarAngle = Math.PI * 0.49;

const ground = new THREE.Mesh(new THREE.PlaneGeometry(4e6, 4e6), new THREE.MeshLambertMaterial({ color: '#eee9dc' }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = -2;
scene.add(ground);

// Keep the canvas exactly the size of its box, including when the side sheet opens.
function fit() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(fit).observe(container);
fit();

// ---------------------------------------------------------------- camera flights (smooth, cancelled by the user)
let flight = null;
function flyTo(target, position, ms = 1300) {
  flight = {
    from: { t: controls.target.clone(), p: camera.position.clone() },
    to: { t: target.clone(), p: position.clone() },
    start: performance.now(), ms,
  };
}
function stepFlight(now) {
  if (!flight) return;
  const k = Math.min(1, (now - flight.start) / flight.ms);
  const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;   // easeInOutCubic
  controls.target.lerpVectors(flight.from.t, flight.to.t, e);
  camera.position.lerpVectors(flight.from.p, flight.to.p, e);
  if (k >= 1) flight = null;
}
controls.addEventListener('start', () => { flight = null; });

function homeView() {
  const [cx, cz] = project(HOME.x, HOME.lat);
  return { t: new THREE.Vector3(cx, 0, cz), p: new THREE.Vector3(cx + 60000, 900000, cz + 650000) };
}

// ---------------------------------------------------------------- colours
const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
const MEAS0 = hex('#cfd8e3'), MEAS1 = hex('#2f4d6b');
const EST0 = hex('#f0c77a'), EST1 = hex('#a0622a');
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

// ---------------------------------------------------------------- tile geometry
// Binary format: see build_tiles.py (ASM1; uint16 vertex count, uint16 height in dm, uint8 flag, quantised outline).
function decodeTile(buf, meta, tint = null) {
  const dv = new DataView(buf);
  const count = dv.getUint32(4, true);
  const T = meta.tile_deg;
  const ox = meta.ox, oy = meta.oy;

  // pass 1: total outline vertices, so typed arrays are sized once (keeps memory low)
  let totalN = 0, scan = 8;
  for (let b = 0; b < count; b++) {
    const n = dv.getUint16(scan, true);
    totalN += n;
    scan += 5 + 4 * n;
  }
  const maxVerts = 5 * totalN;             // n roof vertices + 4 per wall edge
  const maxIdx = 9 * totalN;               // 6 per wall edge + 3 per roof triangle (<= n-2)
  const pos = new Float32Array(maxVerts * 3);
  const col = new Float32Array(maxVerts * 3);
  const idx = new Uint32Array(maxIdx);
  let vp = 0, ip = 0, base = 0;
  let off = 8;

  for (let b = 0; b < count; b++) {
    const n = dv.getUint16(off, true);
    const h = dv.getUint16(off + 2, true) / 10;
    const est = dv.getUint8(off + 4);
    off += 5;
    const xs = new Array(n), zs = new Array(n);
    for (let k = 0; k < n; k++) {
      const qx = dv.getUint16(off, true), qy = dv.getUint16(off + 2, true);
      off += 4;
      const [x, z] = project(ox + qx / 65535 * T, oy + qy / 65535 * T);
      xs[k] = x; zs[k] = z;
    }
    const t = Math.min(h / 30, 1);
    const c = tint || (est ? lerp3(EST0, EST1, t) : lerp3(MEAS0, MEAS1, t));
    const wc = [c[0] * 0.82, c[1] * 0.82, c[2] * 0.82];

    // roof: triangulated outline at height h
    let tris = [];
    try {
      tris = THREE.ShapeUtils.triangulateShape(xs.map((x, i) => new THREE.Vector2(x, zs[i])), []);
    } catch (e) { tris = []; }
    for (let k = 0; k < n; k++) {
      pos[vp * 3] = xs[k]; pos[vp * 3 + 1] = h; pos[vp * 3 + 2] = zs[k];
      col[vp * 3] = c[0]; col[vp * 3 + 1] = c[1]; col[vp * 3 + 2] = c[2];
      vp++;
    }
    for (const tr of tris) { idx[ip++] = base + tr[0]; idx[ip++] = base + tr[1]; idx[ip++] = base + tr[2]; }
    base += n;

    // walls: one quad per edge (flat-shaded in the material, so no normals are stored)
    for (let k = 0; k < n; k++) {
      const j = (k + 1) % n;
      const x1 = xs[k], z1 = zs[k], x2 = xs[j], z2 = zs[j];
      const w = base;
      const quad = [x1, 0, z1, x2, 0, z2, x1, h, z1, x2, h, z2];
      for (let q = 0; q < 4; q++) {
        pos[vp * 3] = quad[q * 3]; pos[vp * 3 + 1] = quad[q * 3 + 1]; pos[vp * 3 + 2] = quad[q * 3 + 2];
        col[vp * 3] = wc[0]; col[vp * 3 + 1] = wc[1]; col[vp * 3 + 2] = wc[2];
        vp++;
      }
      idx[ip++] = w; idx[ip++] = w + 1; idx[ip++] = w + 3;
      idx[ip++] = w; idx[ip++] = w + 3; idx[ip++] = w + 2;
      base += 4;
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, vp * 3), 3));
  geom.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, vp * 3), 3));
  geom.setIndex(new THREE.BufferAttribute(idx.subarray(0, ip), 1));
  geom.computeBoundingSphere();
  return { geom, count };
}

const material = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide });

// ---------------------------------------------------------------- state
let INDEX = null;
const loaded = new Map();      // key -> { mesh, count, buf, meta }
const loading = new Set();
let overviewMesh = null;
const overviewSlot = new Map();
const tileList = [];
const keyOf = (t) => `${t.tx}_${t.ty}`;

function setLoad(pct, text) {
  $('loadbar').style.width = pct + '%';
  if (text) $('loadtext').textContent = text;
}

async function init() {
  setLoad(8, 'Reading the tile index…');
  const r = await fetch(BASE + 'index.json');
  if (!r.ok) throw new Error('tile index ' + r.status);
  INDEX = await r.json();
  $('st-total').textContent = fmtInt(INDEX.total_buildings);
  $('st-est').textContent = fmtInt(INDEX.estimated_height_buildings);
  $('st-tiles-all').textContent = fmtInt(INDEX.tiles.length);

  setLoad(30, 'Building the state overview…');
  await new Promise((res) => setTimeout(res, 0));   // let the loader paint
  const maxN = Math.max(...INDEX.tiles.map((t) => t.n));
  for (const t of INDEX.tiles) {
    const [x, z] = project(t.lon, t.lat);
    tileList.push({ key: keyOf(t), x, z, n: t.n, file: t.file, tx: t.tx, ty: t.ty, lon: t.lon, lat: t.lat });
  }
  // overview: one box per tile, height and colour by building count (density, not real heights)
  const boxGeom = new THREE.BoxGeometry(1, 1, 1);
  boxGeom.translate(0, 0.5, 0);
  overviewMesh = new THREE.InstancedMesh(boxGeom, new THREE.MeshLambertMaterial(), tileList.length);
  overviewMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const m = new THREE.Matrix4(), p = new THREE.Vector3(), s = new THREE.Vector3(), q = new THREE.Quaternion();
  const tileW = INDEX.tile_deg * KX, tileD = INDEX.tile_deg * KY;
  const col = new THREE.Color();
  tileList.forEach((t, i) => {
    t.h = 60 + 2400 * Math.log10(1 + t.n) / Math.log10(1 + maxN);
    p.set(t.x, 0, t.z); s.set(tileW * 0.92, t.h, tileD * 0.92);
    m.compose(p, q, s);
    overviewMesh.setMatrixAt(i, m);
    const f = Math.log10(1 + t.n) / Math.log10(1 + maxN);
    col.setRGB(...lerp3(MEAS0, MEAS1, f));
    overviewMesh.setColorAt(i, col);
    overviewSlot.set(t.key, i);
  });
  overviewMesh.instanceColor.needsUpdate = true;
  scene.add(overviewMesh);

  setLoad(80, 'Drawing the state boundary…');
  const b = await (await fetch('/static/assam3d/assam_boundary.geojson')).json();
  addOutline(b.features[0].geometry);

  setLoad(100, 'Ready');
  // intro: fly down from high above the state to the home view, then hand over to the user
  const [cx, cz] = project(HOME.x, HOME.lat);
  controls.target.set(cx, 0, cz);
  camera.position.set(cx + 5000, 1100000, cz + 20000);
  controls.update();
  const h = homeView();
  flyTo(h.t, h.p, 2800);
  setTimeout(() => $('loader').classList.add('done'), 350);
  setStatus('Overview of the state. Zoom in to load building outlines.');
}

function addOutline(geom) {
  const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
  const pts = [];
  for (const poly of polys) {
    const ring = poly[0];
    for (let i = 0; i < ring.length - 1; i++) {
      const a = project(ring[i][0], ring[i][1]), b = project(ring[i + 1][0], ring[i + 1][1]);
      pts.push(a[0], 3, a[1], b[0], 3, b[1]);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: '#b7791f' })));
}

// ---------------------------------------------------------------- streaming
const queue = [];
let inFlight = 0;

function setOverviewVisible(key, visible) {
  const i = overviewSlot.get(key);
  if (i === undefined) return;
  const t = tileList[i];
  const tileW = INDEX.tile_deg * KX, tileD = INDEX.tile_deg * KY;
  const m = new THREE.Matrix4();
  const s = visible ? new THREE.Vector3(tileW * 0.92, t.h, tileD * 0.92) : new THREE.Vector3(0, 0, 0);
  m.compose(new THREE.Vector3(t.x, 0, t.z), new THREE.Quaternion(), s);
  overviewMesh.setMatrixAt(i, m);
  overviewMesh.instanceMatrix.needsUpdate = true;
}

async function loadTile(t) {
  inFlight++;
  loading.add(t.key);
  try {
    const buf = await (await fetch(BASE + t.file)).arrayBuffer();
    const meta = { tile_deg: INDEX.tile_deg, ox: INDEX.origin[0] + t.tx * INDEX.tile_deg, oy: INDEX.origin[1] + t.ty * INDEX.tile_deg };
    const { geom, count } = decodeTile(buf, meta);
    const mesh = new THREE.Mesh(geom, material);
    scene.add(mesh);
    loaded.set(t.key, { mesh, count, buf, meta });
    setOverviewVisible(t.key, false);
  } catch (e) {
    console.error('tile failed', t.file, e);
  } finally {
    inFlight--;
    loading.delete(t.key);
    pump();
  }
}

function pump() {
  while (inFlight < CONCURRENCY && queue.length) {
    const t = queue.shift();
    if (!loaded.has(t.key) && !loading.has(t.key)) loadTile(t);
  }
}

function unloadTile(key) {
  const e = loaded.get(key);
  if (!e) return;
  scene.remove(e.mesh);
  e.mesh.geometry.dispose();
  loaded.delete(key);
  setOverviewVisible(key, true);
}

function setStatus(text) { $('status').textContent = text; }

let lastUpdate = 0;
function refreshDetail(force) {
  if (!INDEX) return;
  const now = performance.now();
  if (!force && now - lastUpdate < 250) return;
  lastUpdate = now;
  const camDist = camera.position.distanceTo(controls.target);
  const want = [];
  if (camDist < DETAIL_MAX_DIST) {
    const tx = controls.target.x, tz = controls.target.z;
    const radius = Math.max(9000, camDist * 0.9 + 6000);
    const near = tileList
      .map((t) => ({ t, d: Math.hypot(t.x - tx, t.z - tz) }))
      .filter((o) => o.d < radius)
      .sort((a, b) => a.d - b.d);
    let budget = 0;
    for (const o of near) {
      if (want.length && budget + o.t.n > detailBudget) break;
      want.push(o.t);
      budget += o.t.n;
    }
  }
  const wantKeys = new Set(want.map((t) => t.key));
  for (const key of [...loaded.keys()]) if (!wantKeys.has(key)) unloadTile(key);
  queue.length = 0;
  for (const t of want) if (!loaded.has(t.key) && !loading.has(t.key)) queue.push(t);
  pump();
  let buildings = 0;
  loaded.forEach((e) => { buildings += e.count; });
  $('st-tiles').textContent = fmtInt(loaded.size);
  if (queue.length || inFlight) setStatus(`Loading ${queue.length + inFlight} tile(s)…`);
  else if (loaded.size) setStatus(`${fmtInt(buildings)} building outlines in view. Click one to predict its grade.`);
  else setStatus('Overview of the state. Zoom in to load building outlines.');
}

controls.addEventListener('change', () => refreshDetail(false));
setInterval(() => refreshDetail(true), 700);

// ---------------------------------------------------------------- picking a building
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const hitPoint = new THREE.Vector3();
let downAt = null;

renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt) return;
  const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
  downAt = null;
  if (moved > 5) return;                       // it was a drag, not a click
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(groundPlane, hitPoint)) return;
  pickAt(hitPoint.x, hitPoint.z);
});

// Find the building whose footprint contains the ground point (x, z), in the loaded tiles.
function pickAt(x, z) {
  const [lon, lat] = unproject(x, z);
  const tx = Math.floor((lon - INDEX.origin[0]) / INDEX.tile_deg);
  const ty = Math.floor((lat - INDEX.origin[1]) / INDEX.tile_deg);
  const entry = loaded.get(`${tx}_${ty}`);
  if (!entry) { showMessage('Not loaded yet', 'Zoom in closer so this area’s outlines load, then click again.'); return; }
  const found = findInTile(entry, x, z);
  if (!found) { showMessage('No building here', 'Nothing under the cursor. Click on a building’s roof or footprint.'); return; }
  selectBuilding(entry, found);
}

function pointInPoly(px, pz, xs, zs) {
  let inside = false;
  for (let i = 0, j = xs.length - 1; i < xs.length; j = i++) {
    if ((zs[i] > pz) !== (zs[j] > pz) && px < (xs[j] - xs[i]) * (pz - zs[i]) / (zs[j] - zs[i]) + xs[i]) inside = !inside;
  }
  return inside;
}

function findInTile(entry, px, pz) {
  const dv = new DataView(entry.buf);
  const count = dv.getUint32(4, true);
  const { ox, oy } = entry.meta, T = entry.meta.tile_deg;
  let off = 8;
  for (let b = 0; b < count; b++) {
    const n = dv.getUint16(off, true);
    const hdm = dv.getUint16(off + 2, true);
    const est = dv.getUint8(off + 4);
    const start = off;
    off += 5;
    const xs = new Array(n), zs = new Array(n);
    for (let k = 0; k < n; k++) {
      const qx = dv.getUint16(off, true), qy = dv.getUint16(off + 2, true);
      off += 4;
      [xs[k], zs[k]] = project(ox + qx / 65535 * T, oy + qy / 65535 * T);
    }
    if (pointInPoly(px, pz, xs, zs)) {
      return { start, end: off, n, height: hdm / 10, est, xs, zs };
    }
  }
  return null;
}

// ---------------------------------------------------------------- selected building, panel and label
let highlight = null;
let selected = null;               // { cx, cz, height, area, lat, lon, est }
let lastGrade = null;
const outlineMat = new THREE.LineBasicMaterial({ color: '#14213d' });
let formControls = null;

const predictor = createPredictor({
  onLoading: (on) => $('bspin').classList.toggle('on', on),
  onResult: (out) => showPrediction(out),
  onError: (msg) => { $('bbadge-v').textContent = 'Could not predict: ' + msg; },
});

function syncOpen() {
  $('app').classList.toggle('open', !$('bpanel').hidden || !$('epanel').hidden);
  $('example-btn').setAttribute('aria-expanded', String(!$('epanel').hidden));
}

function openPanel() {
  $('epanel').hidden = true;
  $('bpanel').hidden = false;
  syncOpen();
}

function closePanel() {
  $('bpanel').hidden = true;
  clearSelection();
  syncOpen();
}

function clearSelection() {
  if (highlight) { scene.remove(highlight); highlight.geometry.dispose(); highlight = null; }
  selected = null;
  lastGrade = null;
  $('label').hidden = true;
}

function selectBuilding(entry, found) {
  // area from the shoelace formula, in square metres
  let a2 = 0;
  for (let i = 0; i < found.n; i++) {
    const j = (i + 1) % found.n;
    a2 += found.xs[i] * found.zs[j] - found.xs[j] * found.zs[i];
  }
  const area = Math.abs(a2) / 2;
  const cx = found.xs.reduce((s, v) => s + v, 0) / found.n;
  const cz = found.zs.reduce((s, v) => s + v, 0) / found.n;
  const [lon, lat] = unproject(cx, cz);
  clearSelection();
  selected = { cx, cz, area, lon, lat, height: found.height, est: found.est };

  // highlight: rebuild just this building as its own mesh, neutral until a grade arrives
  const body = new Uint8Array(8 + (found.end - found.start));
  body.set(new TextEncoder().encode('ASM1'), 0);
  new DataView(body.buffer).setUint32(4, 1, true);
  body.set(new Uint8Array(entry.buf, found.start, found.end - found.start), 8);
  const { geom } = decodeTile(body.buffer, entry.meta, PENDING_RGB);
  highlight = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide }));
  highlight.position.y = 0.4;
  scene.add(highlight);
  const ring = found.xs.flatMap((x, i) => [x, found.height + 0.4, found.zs[i], found.xs[(i + 1) % found.n], found.height + 0.4, found.zs[(i + 1) % found.n]]);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(ring, 3));
  highlight.add(new THREE.LineSegments(lg, outlineMat));

  // fly in to the building, from a 3/4 angle
  const span = Math.sqrt(Math.max(area, 20));
  const dist = Math.min(1400, Math.max(260, span * 2.6 + found.height * 2));
  const target = new THREE.Vector3(cx, found.height * 0.45, cz);
  const dir = new THREE.Vector3(0.55, 0.6, 0.58).normalize();
  flyTo(target, target.clone().addScaledVector(dir, dist), 1100);

  $('btitle').textContent = `${area < 1000 ? area.toFixed(0) : fmtInt(Math.round(area))} m² building`;
  $('bfacts').innerHTML =
    `<div><b>${area < 1000 ? area.toFixed(0) : fmtInt(Math.round(area))} m²</b><span>footprint</span></div>` +
    `<div><b>${found.height.toFixed(1)} m</b><span>${found.est ? 'height estimated' : 'height measured'}</span></div>` +
    `<div><b>${lat.toFixed(4)}, ${lon.toFixed(4)}</b><span>lat, lon</span></div>`;
  $('bbadge').hidden = false;
  $('bbadge').style.background = PENDING_HEX;
  $('bbadge-v').textContent = 'Calculating…';
  $('bbars').innerHTML = '';
  $('inputs').hidden = false;
  $('label-v').textContent = 'Predicting…';
  $('label').style.setProperty('--c', PENDING_HEX);
  $('label').hidden = false;
  openPanel();
  if (formControls) predictor.now(formControls.read());
}

function showPrediction(out) {
  if (!selected) return;
  const g = Number(out.damage_grade);
  lastGrade = g;
  const meta = GRADES[g];
  $('bbadge').style.background = meta.hex;
  $('bbadge-v').textContent = `${g} · ${meta.label}`;
  renderProbBars($('bbars'), out.probabilities);
  $('label-v').textContent = `${g} · ${meta.label}`;
  $('label').style.setProperty('--c', meta.hex);
  if (highlight) {
    const c = new THREE.Color(...meta.rgb);
    const arr = highlight.geometry.attributes.color.array;
    for (let i = 0; i < arr.length; i += 3) { arr[i] = c.r; arr[i + 1] = c.g; arr[i + 2] = c.b; }
    highlight.geometry.attributes.color.needsUpdate = true;
  }
}

function showMessage(title, msg) {
  clearSelection();
  $('btitle').textContent = title;
  $('bfacts').innerHTML = `<p class="pmsg">${msg}</p>`;
  $('bbadge').hidden = true;
  $('bbars').innerHTML = '';
  $('inputs').hidden = true;
  openPanel();
}

const anchor = new THREE.Vector3();
function updateLabel() {
  const el = $('label');
  if (!selected) { el.hidden = true; return; }
  anchor.set(selected.cx, selected.height + 6, selected.cz).project(camera);
  if (anchor.z > 1 || Math.abs(anchor.x) > 1.2 || Math.abs(anchor.y) > 1.2) { el.style.opacity = '0'; return; }
  const w = container.clientWidth, h = container.clientHeight;
  el.style.opacity = '1';
  el.style.transform = `translate(${((anchor.x + 1) / 2 * w).toFixed(1)}px, ${((1 - anchor.y) / 2 * h).toFixed(1)}px) translate(-50%, -100%) translateY(-10px)`;
}

$('bclose').addEventListener('click', closePanel);

// ---------------------------------------------------------------- example: model vs real records
const ex = { seed: 0, busy: false };
const GRADE_TEXT = { 1: 'Low', 2: 'Medium', 3: 'Near-complete' };
const fmtPct = (v, d = 1) => (v == null ? '–' : (v * 100).toFixed(d) + '%');
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const gradeChip = (g) => `<span class="grade" style="background:${GRADES[g].hex}">${g}</span>`;

function openExamples() {
  clearSelection();
  $('bpanel').hidden = true;
  $('epanel').hidden = false;
  syncOpen();
  setActiveCity(null);
  // the examples are survey records, so the camera goes to the city view rather than a building
  const x = project(91.7362, 26.1445)[0], z = project(91.7362, 26.1445)[1];
  flyTo(new THREE.Vector3(x, 0, z), new THREE.Vector3(x + 2500, 4500, z + 4000), 1500);
  runExamples(ex.seed === 0);
}

function closeExamples() {
  $('epanel').hidden = true;
  syncOpen();
}

async function runExamples(fresh) {
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

function renderExamples(d) {
  $('e-acc').textContent = fmtPct(d.accuracy);
  $('e-full').textContent = fmtPct(d.test_accuracy, 2);
  $('e-maj').textContent = fmtPct(d.majority_accuracy);
  $('e-status').textContent = `Sample ${d.correct} of ${d.n} correct. The model is ${d.correct === d.n ? 'right on every record here' : `off on ${d.n - d.correct} here`}. Small samples vary a lot; the full held-out figure is the reliable one.`;

  // confusion matrix: rows are the real grade, columns the predicted grade
  const head = `<tr><th>Real ↓ · Predicted →</th>${[1, 2, 3].map((g) => `<th>${gradeChip(g)} ${GRADE_TEXT[g]}</th>`).join('')}<th>Recall</th></tr>`;
  const body = d.confusion.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    const cells = row.map((c, j) => `<td class="${i === j ? 'diag' : ''}">${c}</td>`).join('');
    return `<tr><th>${gradeChip(i + 1)} ${GRADE_TEXT[i + 1]}</th>${cells}<td>${total ? fmtPct(row[i] / total, 0) : '–'}</td></tr>`;
  }).join('');
  $('e-cm').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;

  // per-record table
  const rows = d.rows.map((r) => {
    const inp = r.inputs;
    const walls = inp.walls.length ? inp.walls.map(esc).join(', ') : 'none listed';
    return `<tr>
      <td>${esc(inp.floors)} fl · ${esc(inp.age)} yr · ${esc(walls)}</td>
      <td>${gradeChip(r.actual)}</td>
      <td>${gradeChip(r.predicted)}</td>
      <td>${fmtPct(r.confidence, 0)}</td>
      <td class="${r.correct ? 'ok' : 'miss'}">${r.correct ? '✓ match' : '✗ miss'}</td>
    </tr>`;
  }).join('');
  $('e-rows').innerHTML = `<thead><tr><th>Survey inputs</th><th>Real</th><th>Predicted</th><th>Confidence</th><th>Result</th></tr></thead><tbody>${rows}</tbody>`;
  $('e-note').textContent = d.note;
}

$('example-btn').addEventListener('click', () => {
  if ($('epanel').hidden) openExamples(); else closeExamples();
});
$('eclose').addEventListener('click', closeExamples);
$('e-new').addEventListener('click', () => runExamples(true));
$('e-n').addEventListener('change', () => runExamples(false));
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (!$('bpanel').hidden) closePanel();
  else if (!$('epanel').hidden) closeExamples();
});

loadOptions().then((options) => {
  formControls = buildControls($('bform'), options, (payload) => {
    if (selected) predictor.schedule(payload);
  });
}).catch((e) => { $('bfacts').innerHTML = `<p class="pmsg">Model unavailable: ${e.message}</p>`; });

// ---------------------------------------------------------------- city dock
const cityButtons = [...document.querySelectorAll('.city[data-lat]')];
function setActiveCity(btn) {
  cityButtons.forEach((b) => b.classList.toggle('on', b === btn));
  $('reset-view').classList.toggle('on', !btn);
}
cityButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const [x, z] = project(Number(btn.dataset.lon), Number(btn.dataset.lat));
    const t = new THREE.Vector3(x, 0, z);
    flyTo(t, new THREE.Vector3(x + 2500, 4500, z + 4000), 1500);
    setActiveCity(btn);
    setTimeout(() => refreshDetail(true), 200);
  });
});
$('reset-view').addEventListener('click', () => {
  const h = homeView();
  flyTo(h.t, h.p, 1600);
  setActiveCity(null);
  setTimeout(() => refreshDetail(true), 200);
});

// ---------------------------------------------------------------- loop
let lastW = 0, lastH = 0;
let lastFrame = performance.now(), frameAvg = 16, lastAdapt = 0;
renderer.setAnimationLoop((now) => {
  // resize check every frame too, so the canvas follows the side sheet even if the observer is late
  const w = container.clientWidth, h = container.clientHeight;
  if (w !== lastW || h !== lastH) { lastW = w; lastH = h; fit(); }

  // adaptive detail: shrink the building budget when frames run long, grow it back when there is headroom
  const dt = now - lastFrame;
  lastFrame = now;
  frameAvg = frameAvg * 0.95 + Math.min(dt, 500) * 0.05;
  if (now - lastAdapt > 2000 && INDEX) {
    lastAdapt = now;
    if (frameAvg > 40 && detailBudget > DETAIL_MIN) { detailBudget = Math.max(DETAIL_MIN, Math.round(detailBudget * 0.6)); refreshDetail(true); }
    else if (frameAvg < 22 && detailBudget < DETAIL_MAX) { detailBudget = Math.min(DETAIL_MAX, Math.round(detailBudget * 1.25)); refreshDetail(true); }
  }
  stepFlight(now);
  controls.update();
  updateLabel();
  renderer.render(scene, camera);
});

init().catch((e) => {
  console.error(e);
  setLoad(100, 'Could not load the tile index.');
  setStatus('Could not load the map: ' + e.message);
  $('loader').classList.add('done');
});
