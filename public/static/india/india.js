// India in 3D: one map for the whole country.
//
// Four layers, each from a real record:
//   states   36 state and union-territory outlines, built from the Survey of India
//            boundary set (public/static/india/states.json).
//   hazard   the national 0.25 degree grid: Gutenberg-Richter energy per cell, smoothed
//            and split into terciles (GET /api/india/hazard).
//   events   the catalogued M4+ epicentres that fall inside an Indian outline
//            (GET /api/india/events).
//   buildings Assam's 13.3 million Overture footprints, streamed as 0.1 degree tiles,
//            exactly as the previous page loaded them. Assam is the only state with
//            packaged footprints, so it is the only state drawn at building level.
//
// Click a state for its earthquake record and to score a building profile; zoom into
// Assam and click a building for its damage grade.
import * as THREE from 'three';
import { OrbitControls } from '/static/assam3d/vendor/OrbitControls.js';
import { GRADES, SHARED_CSS, loadOptions, buildControls, createPredictor, renderProbBars } from '/static/shared/shared.js';

document.getElementById('shared-css').textContent = SHARED_CSS;

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n).toLocaleString('en-US');

// ---------------------------------------------------------------- projection
// Same convention as the Assam tiles: metres east/north of a reference point, x = lon,
// z = -lat, so the existing tile payloads decode unchanged.
const REF_LON = 82.0, REF_LAT = 22.0;
const KX = 111320 * Math.cos(REF_LAT * Math.PI / 180), KY = 110574;
const project = (lon, lat) => [(lon - REF_LON) * KX, -(lat - REF_LAT) * KY];
const unproject = (x, z) => [REF_LON + x / KX, REF_LAT - z / KY];

const TILES = '/static/assam3d/tiles/';
const ASSAM_BBOX = [89.7, 24.14, 96.02, 27.98];
const DETAIL_MAX_DIST = 320000;        // metres: above this, no building tiles
const OVERVIEW_VISIBLE = 620000;       // metres: above this, only the national layers
const DETAIL_MAX = 160000;             // most building outlines in memory at once
const DETAIL_MIN = 40000;
let detailBudget = DETAIL_MAX;
const CONCURRENCY = 3;
const PENDING_RGB = [0.48, 0.52, 0.60];
const PENDING_HEX = '#7a8699';

const HAZARD_HEX = { r: '#c53030', y: '#d69e2e', g: '#2f855a' };
const MAG_HEX = (m) => (m >= 7 ? '#d64545' : m >= 6 ? '#e08a2c' : '#c9a227');

// ---------------------------------------------------------------- renderer
const container = $('scene');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);
} catch (err) {
  const card = document.querySelector('.lcard');
  if (card) {
    card.innerHTML = '<div class="ltitle">3D needs WebGL</div>'
      + '<div class="lsub">This browser or GPU has WebGL disabled, so the map cannot be drawn. '
      + 'The counts, the Example panel and the <a href="/2d" style="color:#f6d28a">2D world risk map</a> still work.</div>';
    const box = document.getElementById('loader');
    if (box) box.classList.add('nowebgl');
  }
  window.__noGL = true;
  $('status').textContent = 'WebGL unavailable — the 2D pages still work.';
  fetch('/api/india/status').then((r) => r.json()).then((s) => {
    $('st-states').textContent = s.states;
    $('st-events').textContent = fmtInt(s.events);
    $('st-cells').textContent = fmtInt(s.cells);
    $('st-buildings').textContent = fmtInt(s.footprint_buildings);
  }).catch(() => {});
  throw err;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color('#dfe8ef');
scene.fog = new THREE.Fog('#dfe8ef', 4200000, 9000000);

const camera = new THREE.PerspectiveCamera(40, 1, 40, 1.2e7);
scene.add(new THREE.HemisphereLight('#ffffff', '#9aa6b4', 1.7));
const sun = new THREE.DirectionalLight('#ffffff', 1.15);
sun.position.set(-1, 2, 1);
scene.add(sun);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 400;
controls.maxDistance = 6500000;
controls.maxPolarAngle = Math.PI * 0.49;

const ground = new THREE.Mesh(new THREE.PlaneGeometry(1.6e7, 1.6e7),
  new THREE.MeshLambertMaterial({ color: '#eee9dc' }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = -30;
scene.add(ground);

function fit() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(fit).observe(container);
fit();

// ---------------------------------------------------------------- camera flights
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
  const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
  controls.target.lerpVectors(flight.from.t, flight.to.t, e);
  camera.position.lerpVectors(flight.from.p, flight.to.p, e);
  if (k >= 1) flight = null;
}
controls.addEventListener('start', () => { flight = null; });

// Whole-country view: high and slightly tilted, so the country fills the frame.
function countryView() {
  return {
    t: new THREE.Vector3(0, 0, 0),
    p: new THREE.Vector3(0, 4200000, 1450000),
  };
}

// A state's own view: centre on its bbox and pull back until the box fits.
function stateView(s) {
  const [w, so, e, n] = s.bbox;
  const [x, z] = project((w + e) / 2, (so + n) / 2);
  const spanX = Math.abs(e - w) * KX, spanZ = Math.abs(n - so) * KY;
  const tan = Math.tan((camera.fov * Math.PI) / 360);
  const d = Math.max(spanZ / (2 * tan), spanX / (2 * tan) / Math.max(0.6, camera.aspect), 24000) * 1.2;
  return {
    t: new THREE.Vector3(x, 0, z),
    p: new THREE.Vector3(x + d * 0.12, d * 1.02, z + d * 0.42),
  };
}

function setStatus(text) { $('status').textContent = text; }
function setLoad(pct, text) {
  $('loadbar').style.width = pct + '%';
  if (text) $('loadtext').textContent = text;
}

// ---------------------------------------------------------------- data
let STATUS = null;
let STATES = [];
const stateById = new Map();
const layers = { states: true, hazard: true, events: true, buildings: true };

async function loadAll() {
  setLoad(6, 'Reading the state outlines…');
  const [statusRes, statesRes] = await Promise.all([
    fetch('/api/india/status'), fetch('/static/india/states.json'),
  ]);
  if (!statusRes.ok) throw new Error('status ' + statusRes.status);
  if (!statesRes.ok) throw new Error('states ' + statesRes.status);
  STATUS = await statusRes.json();
  STATES = (await statesRes.json()).states;
  for (const s of STATES) {
    const row = STATUS.per_state.find((r) => r.id === s.id) || {};
    s.events = row.events || 0;
    s.events_m5 = row.events_m5 || 0;
    s.largest = row.largest ?? null;
    s.has_footprints = !!row.has_footprints;
    stateById.set(s.id, s);
  }
  $('st-states').textContent = STATUS.states;
  $('st-events').textContent = fmtInt(STATUS.events);
  $('st-cells').textContent = fmtInt(STATUS.cells);
  $('st-buildings').textContent = fmtInt(STATUS.footprint_buildings);

  setLoad(34, 'Drawing the states…');
  buildStates();
  await new Promise((r) => setTimeout(r, 0));

  setLoad(52, 'Drawing the hazard grid…');
  const hazard = await (await fetch('/api/india/hazard')).json();
  buildHazard(hazard.cells, hazard.meta);
  $('st-cells').textContent = fmtInt(hazard.meta.cells_total);
  $('methodLine').textContent = hazard.meta.method;

  setLoad(70, 'Plotting the epicentres…');
  const ev = await (await fetch('/api/india/events?min_mag=4')).json();
  buildEvents(ev.events);
  $('st-events').textContent = fmtInt(ev.count);

  setLoad(84, 'Reading the Assam building tiles…');
  await loadTileIndex();

  setLoad(100, 'Ready');
  const v = countryView();
  controls.target.copy(v.t);
  camera.position.set(0, 7600000, 2600000);
  controls.update();
  flyTo(v.t, v.p, 3000);
  setTimeout(() => $('loader').classList.add('done'), 320);
  setStatus(`${STATUS.states} states and union territories · ${fmtInt(ev.count)} epicentres · ${fmtInt(hazard.meta.cells_total)} hazard cells`);
}

// ---------------------------------------------------------------- states
const stateGroup = new THREE.Group();
scene.add(stateGroup);
const stateMeshes = [];

function stateFill(s) {
  // fill colour by how much of the catalogue sits in the state; Assam is tinted because
  // it is the only one that can be opened at building level
  if (s.has_footprints) return [0.95, 0.87, 0.72];
  const t = Math.min(1, Math.log10(1 + s.events) / Math.log10(1 + 220));
  return [0.95 - 0.11 * t, 0.95 - 0.10 * t, 0.96 - 0.06 * t];
}

function buildStates() {
  const borderPts = [];
  for (const s of STATES) {
    const pos = [], idx = [];
    let base = 0;
    for (const ring of s.rings) {
      const poly = ring.map(([lon, lat]) => {
        const [x, z] = project(lon, lat);
        return new THREE.Vector2(x, z);
      });
      let tris = [];
      try { tris = THREE.ShapeUtils.triangulateShape(poly, []); } catch (e) { tris = []; }
      for (const [lon, lat] of ring) {
        const [x, z] = project(lon, lat);
        pos.push(x, -8, z);
      }
      for (const t of tris) idx.push(base + t[0], base + t[1], base + t[2]);
      base += ring.length;
      for (let i = 0; i + 1 < ring.length; i++) {
        const a = project(ring[i][0], ring[i][1]);
        const b = project(ring[i + 1][0], ring[i + 1][1]);
        borderPts.push(a[0], -6, a[1], b[0], -6, b[1]);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    const c = stateFill(s);
    const mesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({
      color: new THREE.Color(c[0], c[1], c[2]), side: THREE.DoubleSide,
    }));
    mesh.userData = { stateId: s.id, baseColor: c };
    stateGroup.add(mesh);
    stateMeshes.push(mesh);
  }
  const bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.Float32BufferAttribute(borderPts, 3));
  stateGroup.add(new THREE.LineSegments(bg, new THREE.LineBasicMaterial({
    color: '#5d6b7d', transparent: true, opacity: 0.8,
  })));
}

// ---------------------------------------------------------------- hazard grid
let hazardMesh = null, hazardCells = [];
const CELL = 0.25;

function buildHazard(cells, meta) {
  hazardCells = cells;
  const scores = cells.map((c) => c[3]);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const side = CELL * KX;
  const geo = new THREE.BoxGeometry(1, 1, 1);
  geo.translate(0, 0.5, 0);
  hazardMesh = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial({ flatShading: true }), cells.length);
  hazardMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cells.length * 3), 3);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), col = new THREE.Color();
  cells.forEach(([lon, lat, cls, score], i) => {
    const [x, z] = project(lon + CELL / 2, lat + CELL / 2);
    const t = hi > lo ? (score - lo) / (hi - lo) : 0.5;
    // Exaggerated on purpose: at national scale a truthful 20 km column would be a
    // pixel. 20 km to 220 km reads as a spike; the mesh is scaled down as you zoom
    // (see the render loop), so it never swamps the building detail.
    const hgt = 20000 + t * t * 200000;
    m.compose(new THREE.Vector3(x, 0, z), q, new THREE.Vector3(side * 0.94, hgt, side * 0.94));
    hazardMesh.setMatrixAt(i, m);
    col.set(HAZARD_HEX[cls] || '#888888');
    hazardMesh.setColorAt(i, col);
  });
  hazardMesh.instanceMatrix.needsUpdate = true;
  hazardMesh.visible = layers.hazard;
  scene.add(hazardMesh);
  $('lg-hazard').textContent = `${fmtInt(cells.length)} cells · ${CELL}° · red ${meta.thresholds_log10_energy.red_from ?? '–'} / yellow ${meta.thresholds_log10_energy.yellow_from ?? '–'} log10 J`;
}

// ---------------------------------------------------------------- epicentres
// Drawn as points with size attenuation off, so an epicentre is 6 px whether you are
// looking at the whole country or at one state. Colour carries the magnitude.
let eventMesh = null;

function buildEvents(events) {
  const pos = new Float32Array(events.length * 3);
  const col = new Float32Array(events.length * 3);
  const c = new THREE.Color();
  events.forEach((e, i) => {
    const [x, z] = project(e.lon, e.lat);
    pos[i * 3] = x; pos[i * 3 + 1] = 6; pos[i * 3 + 2] = z;
    c.set(MAG_HEX(e.mag));
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.computeBoundingSphere();
  eventMesh = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 6, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false,
  }));
  eventMesh.visible = layers.events;
  scene.add(eventMesh);
}

// ---------------------------------------------------------------- Assam building tiles
// The decode and streaming code is the same as the Assam page; only the camera scale and
// the trigger changed, so footprints appear when you actually zoom into Assam.
let INDEX = null;
const overviewSlot = new Map();
let overviewMesh = null;
const tileList = [];
const loaded = new Map();
const loading = new Set();
const keyOf = (t) => `${t.tx}_${t.ty}`;

function decodeTile(buf, meta, tint = null) {
  const MEAS0 = [0.81, 0.85, 0.89], MEAS1 = [0.18, 0.30, 0.42];
  const EST0 = [0.94, 0.78, 0.48], EST1 = [0.63, 0.38, 0.16];
  const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const dv = new DataView(buf);
  const count = dv.getUint32(4, true);
  const T = meta.tile_deg, ox = meta.ox, oy = meta.oy;
  let totalN = 0, scan = 8;
  for (let b = 0; b < count; b++) {
    const n = dv.getUint16(scan, true);
    totalN += n;
    scan += 5 + 4 * n;
  }
  const maxVerts = 5 * totalN, maxIdx = 9 * totalN;
  const pos = new Float32Array(maxVerts * 3), col = new Float32Array(maxVerts * 3);
  const idx = new Uint32Array(maxIdx);
  let vp = 0, ip = 0, base = 0, off = 8;
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
    for (let k = 0; k < n; k++) {
      const j = (k + 1) % n;
      const x1 = xs[k], z1 = zs[k], x2 = xs[j], z2 = zs[j];
      const w = base;
      const quad = [x1, 0, z1, x2, 0, z2, x1, h, z1, x2, h, z2];
      for (let q2 = 0; q2 < 4; q2++) {
        pos[vp * 3] = quad[q2 * 3]; pos[vp * 3 + 1] = quad[q2 * 3 + 1]; pos[vp * 3 + 2] = quad[q2 * 3 + 2];
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

async function loadTileIndex() {
  const r = await fetch(TILES + 'index.json');
  if (!r.ok) throw new Error('tile index ' + r.status);
  INDEX = await r.json();
  const maxN = Math.max(...INDEX.tiles.map((t) => t.n));
  const box = new THREE.BoxGeometry(1, 1, 1);
  box.translate(0, 1, 0);
  overviewMesh = new THREE.InstancedMesh(box, new THREE.MeshLambertMaterial(), INDEX.tiles.length);
  overviewMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(INDEX.tiles.length * 3), 3);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3();
  const col = new THREE.Color();
  const tileW = INDEX.tile_deg * KX, tileD = INDEX.tile_deg * KY;
  INDEX.tiles.forEach((t, i) => {
    const [x, z] = project(t.lon, t.lat);
    const f = Math.log10(1 + t.n) / Math.log10(1 + maxN);
    const h = 900 + 9000 * f;
    p.set(x, 0, z); s.set(tileW * 0.92, h, tileD * 0.92);
    m.compose(p, q, s);
    overviewMesh.setMatrixAt(i, m);
    col.setRGB(...[0.72 - 0.35 * f, 0.66 - 0.32 * f, 0.55 - 0.28 * f]);
    overviewMesh.setColorAt(i, col);
    tileList.push({ key: keyOf(t), x, z, n: t.n, file: t.file, tx: t.tx, ty: t.ty, lon: t.lon, lat: t.lat, h });
    overviewSlot.set(keyOf(t), i);
  });
  overviewMesh.instanceMatrix.needsUpdate = true;
  overviewMesh.visible = layers.buildings;
  scene.add(overviewMesh);
  $('st-buildings').textContent = fmtInt(INDEX.total_buildings);
}

const queue = [];
let inFlight = 0;

function setOverviewVisible(key, visible) {
  const i = overviewSlot.get(key);
  if (i === undefined || !INDEX) return;
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
    const buf = await (await fetch(TILES + t.file)).arrayBuffer();
    const meta = {
      tile_deg: INDEX.tile_deg,
      ox: INDEX.origin[0] + t.tx * INDEX.tile_deg,
      oy: INDEX.origin[1] + t.ty * INDEX.tile_deg,
    };
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

let lastUpdate = 0;
function refreshDetail(force) {
  if (!INDEX || !layers.buildings) return;
  const now = performance.now();
  if (!force && now - lastUpdate < 250) return;
  lastUpdate = now;
  const camDist = camera.position.distanceTo(controls.target);
  if (overviewMesh) overviewMesh.visible = camDist < OVERVIEW_VISIBLE;
  const [tlon, tlat] = unproject(controls.target.x, controls.target.z);
  const overAssam = tlon > ASSAM_BBOX[0] - 1 && tlon < ASSAM_BBOX[2] + 1
    && tlat > ASSAM_BBOX[1] - 1 && tlat < ASSAM_BBOX[3] + 1;
  const want = [];
  if (camDist < DETAIL_MAX_DIST && overAssam) {
    const tx = controls.target.x, tz = controls.target.z;
    const radius = Math.max(12000, camDist * 0.9 + 6000);
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
  $('st-tiles').textContent = fmtInt(loaded.size);
}

controls.addEventListener('change', () => refreshDetail(false));
setInterval(() => refreshDetail(true), 800);

// ---------------------------------------------------------------- picking
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
  if (moved > 5) return;
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(groundPlane, hitPoint)) return;
  const x = hitPoint.x, z = hitPoint.z;

  // a loaded building wins, so the Assam detail still behaves like the old page
  const [lon, lat] = unproject(x, z);
  if (INDEX) {
    const tx = Math.floor((lon - INDEX.origin[0]) / INDEX.tile_deg);
    const ty = Math.floor((lat - INDEX.origin[1]) / INDEX.tile_deg);
    const entry = loaded.get(`${tx}_${ty}`);
    if (entry) {
      const found = findInTile(entry, x, z);
      if (found) { selectBuilding(entry, found); return; }
    }
  }
  // otherwise: which state is under the cursor?
  if (!layers.states) return;
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(stateMeshes, false);
  if (hits.length) {
    const s = stateById.get(hits[0].object.userData.stateId);
    if (s) selectState(s);
  }
});

function pointInPoly(px, pz, xs, zs) {
  let inside = false;
  for (let i = 0, j = xs.length - 1; i < xs.length; j = i++) {
    if ((zs[i] > pz) !== (zs[j] > pz)
      && px < (xs[j] - xs[i]) * (pz - zs[i]) / (zs[j] - zs[i]) + xs[i]) inside = !inside;
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
    if (pointInPoly(px, pz, xs, zs)) return { start, end: off, n, height: hdm / 10, est, xs, zs };
  }
  return null;
}

// ---------------------------------------------------------------- panels
const predictor = createPredictor({
  onLoading: (on) => $('bspin').classList.toggle('on', on),
  onResult: (out) => showPrediction(out),
  onError: (msg) => { $('bbadge-v').textContent = 'Could not predict: ' + msg; },
});
let formControls = null;
let highlight = null;
let selected = null;
let selectedState = null;
let lastGrade = null;
const outlineMat = new THREE.LineBasicMaterial({ color: '#14213d' });

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
  if (highlight) { scene.remove(highlight); highlight.geometry?.dispose(); highlight = null; }
  if (selectedState) {
    const m = stateMeshes.find((o) => o.userData.stateId === selectedState.id);
    if (m) {
      const c = m.userData.baseColor;
      m.material.color.setRGB(c[0], c[1], c[2]);
    }
    selectedState = null;
  }
  selected = null;
  lastGrade = null;
  $('label').hidden = true;
}

function tintState(s) {
  const m = stateMeshes.find((o) => o.userData.stateId === s.id);
  if (m) m.material.color.set('#e9cf93');
}

function selectState(s) {
  clearSelection();
  selectedState = s;
  tintState(s);
  $('beyebrow').textContent = s.has_footprints ? 'State · building footprints' : 'State';
  $('btitle').textContent = s.name;
  const largest = s.largest ? `M${Number(s.largest).toFixed(1)}` : '–';
  $('bfacts').innerHTML =
    `<div><b>${fmtInt(s.events)}</b><span>M4+ epicentres</span></div>`
    + `<div><b>${fmtInt(s.events_m5)}</b><span>M5+</span></div>`
    + `<div><b>${largest}</b><span>largest since 2000</span></div>`;
  $('bbadge').hidden = false;
  $('bbadge').style.background = PENDING_HEX;
  $('bbadge-v').textContent = 'Calculating…';
  $('bbars').innerHTML = '';
  $('inputs').hidden = false;
  $('stateNote').hidden = false;
  $('stateNote').innerHTML = s.has_footprints
    ? 'Zoom in to reach the building level: Assam has 13,302,299 Overture footprints in this map. Click any building for its own grade.'
    : 'No building footprints are packaged for this state, so the map shows its earthquake record rather than invented buildings.'
      + ' The grade below comes from the building profile you set, not from coordinates.';
  $('bigList').innerHTML = '';
  $('label').hidden = true;
  openPanel();
  if (formControls) predictor.now(formControls.read());
  const v = stateView(s);
  flyTo(v.t, v.p, 1500);
}

function selectBuilding(entry, found) {
  let a2 = 0;
  for (let i = 0; i < found.n; i++) {
    const j = (i + 1) % found.n;
    a2 += found.xs[i] * found.zs[j] - found.xs[j] * found.zs[i];
  }
  const area = Math.abs(a2) / 2;
  const cx = found.xs.reduce((s, v) => s + v, 0) / found.n;
  const cz = found.zs.reduce((s, v) => s + v, 0) / found.n;
  const [lon, lat] = unproject(cx, cz);
  const host = STATES.find((s) => s.has_footprints) || { name: 'Assam', id: 'assam' };
  clearSelection();
  selected = { cx, cz, area, lon, lat, height: found.height, est: found.est };

  const body = new Uint8Array(8 + (found.end - found.start));
  body.set(new TextEncoder().encode('ASM1'), 0);
  new DataView(body.buffer).setUint32(4, 1, true);
  body.set(new Uint8Array(entry.buf, found.start, found.end - found.start), 8);
  const { geom } = decodeTile(body.buffer, entry.meta, PENDING_RGB);
  highlight = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
    vertexColors: true, flatShading: true, side: THREE.DoubleSide,
  }));
  highlight.position.y = 0.4;
  scene.add(highlight);
  const ring = found.xs.flatMap((x, i) => [x, found.height + 0.4, found.zs[i], found.xs[(i + 1) % found.n], found.height + 0.4, found.zs[(i + 1) % found.n]]);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(ring, 3));
  highlight.add(new THREE.LineSegments(lg, outlineMat));

  const span = Math.sqrt(Math.max(area, 20));
  const dist = Math.min(1400, Math.max(260, span * 2.6 + found.height * 2));
  const target = new THREE.Vector3(cx, found.height * 0.45, cz);
  const dir = new THREE.Vector3(0.55, 0.6, 0.58).normalize();
  flyTo(target, target.clone().addScaledVector(dir, dist), 1100);

  $('beyebrow').textContent = `Building · ${host.name}`;
  $('btitle').textContent = `${area < 1000 ? area.toFixed(0) : fmtInt(Math.round(area))} m² building`;
  $('bfacts').innerHTML =
    `<div><b>${area < 1000 ? area.toFixed(0) : fmtInt(Math.round(area))} m²</b><span>footprint</span></div>`
    + `<div><b>${found.height.toFixed(1)} m</b><span>${found.est ? 'height estimated' : 'height measured'}</span></div>`
    + `<div><b>${lat.toFixed(4)}, ${lon.toFixed(4)}</b><span>lat, lon</span></div>`;
  $('stateNote').hidden = true;
  $('bigList').innerHTML = '';
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
  const g = Number(out.damage_grade);
  lastGrade = g;
  const meta = GRADES[g];
  $('bbadge').style.background = meta.hex;
  $('bbadge-v').textContent = `${g} · ${meta.label}`;
  renderProbBars($('bbars'), out.probabilities);
  if (selected) {
    $('label-v').textContent = `${g} · ${meta.label}`;
    $('label').style.setProperty('--c', meta.hex);
    if (highlight) {
      const c = new THREE.Color(...meta.rgb);
      const arr = highlight.geometry.attributes.color.array;
      for (let i = 0; i < arr.length; i += 3) { arr[i] = c.r; arr[i + 1] = c.g; arr[i + 2] = c.b; }
      highlight.geometry.attributes.color.needsUpdate = true;
    }
  }
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

// ---------------------------------------------------------------- layer toggles
// The hazard spikes are exaggerated for the national view. As you zoom towards a state
// they shrink in proportion, so they keep roughly the same apparent height, and below
// 90 km from the ground they step aside entirely — that is building territory.
const HAZARD_SCALE_DIST = 4200000;
function applyScale() {
  const d = camera.position.distanceTo(controls.target);
  const k = Math.min(1, Math.max(0.04, d / HAZARD_SCALE_DIST));
  if (hazardMesh) {
    hazardMesh.scale.y = k;
    hazardMesh.visible = layers.hazard && d > 90000;
  }
  if (eventMesh) {
    eventMesh.visible = layers.events && d > 2500;
    const m = eventMesh.material;
    m.size = d > 400000 ? 6 : 4.5;   // a little tighter once you are inside a state
  }
}

function syncLayers() {
  if (hazardMesh) hazardMesh.visible = layers.hazard;
  if (eventMesh) eventMesh.visible = layers.events;
  stateGroup.visible = layers.states;
  if (overviewMesh) overviewMesh.visible = layers.buildings;
  loaded.forEach((e) => { e.mesh.visible = layers.buildings; });
  for (const [id, on] of Object.entries(layers)) {
    const b = $(`t-${id}`);
    if (b) {
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }
  if (!layers.buildings) {
    for (const key of [...loaded.keys()]) unloadTile(key);
    $('st-tiles').textContent = '0';
  }
  refreshDetail(true);
}
for (const id of ['states', 'hazard', 'events', 'buildings']) {
  $(`t-${id}`).addEventListener('click', () => {
    layers[id] = !layers[id];
    syncLayers();
  });
}

// ---------------------------------------------------------------- state dock and picker
function fillPicker() {
  const sel = $('statePick');
  const order = [...STATES].sort((a, b) => b.events - a.events || a.name.localeCompare(b.name));
  sel.innerHTML = `<option value="">Jump to a state…</option>`
    + order.map((s) => `<option value="${s.id}">${s.name} — ${fmtInt(s.events)} events</option>`).join('');
  sel.addEventListener('change', () => {
    const s = stateById.get(sel.value);
    if (s) { selectState(s); $('reset-view').classList.remove('on'); }
  });
}

document.querySelectorAll('.city[data-state]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const s = stateById.get(btn.dataset.state);
    document.querySelectorAll('.city[data-state]').forEach((b) => b.classList.toggle('on', b === btn));
    $('reset-view').classList.toggle('on', false);
    if (s) selectState(s);
  });
});
$('reset-view').addEventListener('click', () => {
  document.querySelectorAll('.city[data-state]').forEach((b) => b.classList.remove('on'));
  $('statePick').value = '';
  closePanel();
  const v = countryView();
  flyTo(v.t, v.p, 1600);
  setTimeout(() => refreshDetail(true), 200);
});

// ---------------------------------------------------------------- example panel hook
const hooks = {
  focusExamples() {
    if (window.__noGL) return;
    clearSelection();
    const v = countryView();
    flyTo(v.t, v.p, 1400);
  },
};
window.__india3d = hooks;
window.__assam3d = hooks;      // examples.js asked for this name when Assam was the home page

loadOptions().then((options) => {
  formControls = buildControls($('bform'), options, (payload) => {
    if (selected || selectedState) predictor.schedule(payload);
  });
}).catch((e) => {
  $('bfacts').innerHTML = `<p class="pmsg">Model unavailable: ${e.message}</p>`;
});

// ---------------------------------------------------------------- loop
let lastW = 0, lastH = 0, lastFrame = performance.now(), frameAvg = 16, lastAdapt = 0;
renderer.setAnimationLoop((now) => {
  const w = container.clientWidth, h = container.clientHeight;
  if (w !== lastW || h !== lastH) { lastW = w; lastH = h; fit(); }
  const dt = now - lastFrame;
  lastFrame = now;
  frameAvg = frameAvg * 0.95 + Math.min(dt, 500) * 0.05;
  if (now - lastAdapt > 2000 && INDEX) {
    lastAdapt = now;
    if (frameAvg > 40 && detailBudget > DETAIL_MIN) {
      detailBudget = Math.max(DETAIL_MIN, Math.round(detailBudget * 0.6));
      refreshDetail(true);
    } else if (frameAvg < 22 && detailBudget < DETAIL_MAX) {
      detailBudget = Math.min(DETAIL_MAX, Math.round(detailBudget * 1.25));
      refreshDetail(true);
    }
  }
  stepFlight(now);
  controls.update();
  applyScale();
  updateLabel();
  renderer.render(scene, camera);
});

loadAll().then(() => {
  fillPicker();
  syncLayers();
}).catch((e) => {
  console.error(e);
  setLoad(100, 'Could not load the map: ' + e.message);
  setStatus('Could not load the map: ' + e.message);
  $('loader').classList.add('done');
});
