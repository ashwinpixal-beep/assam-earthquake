// Earth in 3D: real country geometry (Natural Earth 1:110m) on a sphere, the global
// M5+/M6+ catalogue as points, a hazard grid from the same catalogue, and a
// click-anywhere damage-grade prediction using the shipped damage model.
import * as THREE from 'three';
import { OrbitControls } from '/static/assam3d/vendor/OrbitControls.js';
import { GRADES, SHARED_CSS, loadOptions, buildControls, createPredictor, renderProbBars } from '/static/shared/shared.js';

document.getElementById('shared-css').textContent = SHARED_CSS;

const R = 100;                       // globe radius in scene units
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString('en-US');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const MAG_BANDS = [
  { min: 5.0, max: 6.0, colour: '#f6d28a', size: 0.55, label: 'M5.0–5.9' },
  { min: 6.0, max: 7.0, colour: '#f0a43a', size: 0.85, label: 'M6.0–6.9' },
  { min: 7.0, max: 10.0, colour: '#e2542c', size: 1.25, label: 'M7.0+' },
];
const HAZARD_COLOUR = { r: '#c53030', y: '#d69e2e', g: '#2f855a' };
let DOT_TEX = null;

// ---------------------------------------------------------------- lat/lon ↔ xyz
// Matches the UV layout of THREE.SphereGeometry with an equirectangular texture,
// so the country texture, the quake points and click handling all agree.
const toVec = (lat, lon, radius = R) => {
  const phi = (90 - lat) * Math.PI / 180;
  const theta = (lon + 180) * Math.PI / 180;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
};
const toLatLon = (v) => {
  const p = v.clone().normalize();
  const lat = 90 - Math.acos(THREE.MathUtils.clamp(p.y, -1, 1)) * 180 / Math.PI;
  const theta = Math.atan2(p.z, -p.x);
  let lon = theta * 180 / Math.PI - 180;
  while (lon < -180) lon += 360;
  while (lon > 180) lon -= 360;
  return [lat, lon];
};

// ---------------------------------------------------------------- scene
const container = $('scene');
let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);
} catch (err) {
  // No WebGL: keep the data side working and say so plainly instead of hanging on the loader.
  const box = document.querySelector('.lcard');
  if (box) {
    box.innerHTML = '<div class="ltitle">3D needs WebGL</div>'
      + '<div class="lsub">This browser or GPU has WebGL disabled, so the globe cannot be drawn. '
      + 'The 2D map shows the same catalogue, the same hazard cells and the same damage model.</div>'
      + '<a class="lgo" href="/world">Open the 2D world map →</a>';
    document.getElementById('loader').classList.add('nowebgl');
  }
  const st = document.getElementById('status');
  if (st) st.textContent = 'WebGL unavailable — showing data only.';
  window.__noGL = true;
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 1, 3000);
camera.position.copy(toVec(20, 80).normalize().multiplyScalar(R * 2.9));

const controls = renderer ? new OrbitControls(camera, renderer.domElement) : null;
if (controls) {
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.rotateSpeed = 0.45;
  controls.zoomSpeed = 0.75;
  controls.enablePan = false;
  controls.minDistance = R * 1.18;
  controls.maxDistance = R * 4.2;
}

scene.add(new THREE.HemisphereLight('#cfe0f2', '#0a1322', 1.05));
const key = new THREE.DirectionalLight('#fff6e6', 1.15);
key.position.set(-1, 0.7, 1.1);
scene.add(key);

const globe = new THREE.Group();
scene.add(globe);

function fit() {
  if (!renderer) return;
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', fit);

// ---------------------------------------------------------------- textures
function drawGraticule(ctx, W, H) {
  ctx.save();
  ctx.strokeStyle = 'rgba(120,160,205,.16)';
  ctx.lineWidth = 1;
  for (let lon = -180; lon <= 180; lon += 30) {
    const x = (lon + 180) / 360 * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = (90 - lat) / 180 * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  ctx.restore();
}

function landTexture(countries, hazard, showHazard, W = 4096, H = 2048) {
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  const ocean = ctx.createLinearGradient(0, 0, 0, H);
  ocean.addColorStop(0, '#0d1c30');
  ocean.addColorStop(0.5, '#122a45');
  ocean.addColorStop(1, '#0d1c30');
  ctx.fillStyle = ocean;
  ctx.fillRect(0, 0, W, H);
  drawGraticule(ctx, W, H);

  const path = (rings) => {
    ctx.beginPath();
    for (const ring of rings) {
      ring.forEach(([lon, lat], i) => {
        const x = (lon + 180) / 360 * W;
        const y = (90 - lat) / 180 * H;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
    }
  };
  const ringArea = (ring) => {
    let a = 0;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      a += (ring[j][0] * ring[i][1]) - (ring[i][0] * ring[j][1]);
    }
    return Math.abs(a / 2);
  };

  // fill outer rings with the land colour, then punch holes with the ocean gradient,
  // so lakes and enclaves (Lesotho, the Caspian) stay water
  const oceanFill = ctx.createLinearGradient(0, 0, 0, H);
  oceanFill.addColorStop(0, '#0d1c30');
  oceanFill.addColorStop(0.5, '#122a45');
  oceanFill.addColorStop(1, '#0d1c30');

  for (const c of countries) {
    const outer = [], holes = [];
    for (const r of c.rings) (ringArea(r) > 12 ? outer : holes).push(r);
    if (!outer.length) continue;
    ctx.fillStyle = '#2a4160';
    path(outer);
    ctx.fill();
    if (holes.length) {
      ctx.fillStyle = oceanFill;
      path(holes);
      ctx.fill();
    }
    ctx.strokeStyle = '#5d7fa6';
    ctx.lineWidth = 2.2;
    ctx.lineJoin = 'round';
    path(c.rings);
    ctx.stroke();
  }

  // hazard cells painted into the same texture: one sphere, no compositing seams
  if (showHazard && hazard && hazard.cells && hazard.cells.length) {
    ctx.save();
    ctx.globalAlpha = hazard.alpha == null ? 0.52 : hazard.alpha;
    const step = hazard.step || 2;
    for (const [lon, lat, cls] of hazard.cells) {
      const x = (lon + 180) / 360 * W;
      const yTop = (90 - (lat + step)) / 180 * H;
      ctx.fillStyle = HAZARD_COLOUR[cls] || '#888888';
      ctx.fillRect(x - 1, yTop - 1, step / 360 * W + 2, step / 180 * H + 2);
    }
    ctx.restore();
  }

  // a soft highlight on the equator-facing side for depth
  const glow = ctx.createRadialGradient(W * 0.5, H * 0.45, H * 0.1, W * 0.5, H * 0.45, H * 0.62);
  glow.addColorStop(0, 'rgba(255,240,210,.07)');
  glow.addColorStop(1, 'rgba(0,0,0,.28)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  if (renderer) tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

// ---------------------------------------------------------------- layers
let eventPoints = [];
let labels = [];
const labelPool = [];
let countriesData = [];
let citiesData = [];

function buildGlobe(countries) {
  applySurface();                     // creates the sphere with the plain land texture
  const sphere = globeSphere;

  // atmosphere
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(R * 1.035, 64, 48),
    new THREE.MeshBasicMaterial({ color: '#6fa8dc', transparent: true, opacity: 0.12, side: THREE.BackSide }),
  );
  globe.add(halo);

  // subtle rim light
  const rim = new THREE.Mesh(
    new THREE.SphereGeometry(R * 1.002, 64, 48),
    new THREE.MeshBasicMaterial({ color: '#0a1322', transparent: true, opacity: 0.18, side: THREE.FrontSide }),
  );
  globe.add(rim);

  globe.userData.pickTarget = sphere;
}

function dotTexture() {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 1, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.55, 'rgba(255,255,255,0.92)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.beginPath(); g.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildEvents(events) {
  for (const p of eventPoints) { globe.remove(p); p.geometry.dispose(); p.material.dispose(); }
  eventPoints = [];
  for (const band of MAG_BANDS) {
    const rows = events.filter((e) => e.mag >= band.min && e.mag < band.max);
    if (!rows.length) continue;
    const pos = new Float32Array(rows.length * 3);
    rows.forEach((e, i) => {
      const v = toVec(e.lat, e.lon, R * 1.004);
      pos[i * 3] = v.x; pos[i * 3 + 1] = v.y; pos[i * 3 + 2] = v.z;
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      color: band.colour, size: band.size * 2.4, sizeAttenuation: true,
      map: DOT_TEX, alphaTest: 0.04, transparent: true, opacity: 0.95, depthWrite: false,
    });
    const pts = new THREE.Points(geom, mat);
    globe.add(pts);
    eventPoints.push(pts);
  }
}

let hazardState = { cells: [], step: 2, on: true, alpha: 0.52, bucket: 0 };
let globeSphere = null;

function applySurface() {
  // rebuild the one and only globe texture with or without the hazard overlay
  const tex = landTexture(countriesData, hazardState, hazardState.on);
  if (!globeSphere) {
    globeSphere = new THREE.Mesh(
      new THREE.SphereGeometry(R, 128, 80),
      new THREE.MeshBasicMaterial({ map: tex }),
    );
    globe.add(globeSphere);
  } else {
    globeSphere.material.map.dispose();
    globeSphere.material.map = tex;
    globeSphere.material.needsUpdate = true;
  }
}

function buildHazard(cells, meta) {
  hazardState.cells = cells || [];
  hazardState.step = (meta && meta.cell_deg) || 2;
  applySurface();
}

// ---------------------------------------------------------------- city labels
function buildLabels(cities) {
  labels = cities.slice(0, 26);
  for (let i = 0; i < labels.length; i += 1) {
    let el = labelPool[i];
    if (!el) {
      el = document.createElement('div');
      el.className = 'citylabel';
      $('labels').appendChild(el);
      labelPool[i] = el;
    }
    el.textContent = labels[i].name;
  }
}

const tmpV = new THREE.Vector3();
function updateLabels() {
  if (!labels.length) return;
  const w = container.clientWidth, h = container.clientHeight;
  const camDir = camera.position.clone().normalize();
  const placed = [];                 // screen boxes already taken this frame
  const cities = labels.map((city, i) => {
    const v = toVec(city.lat, city.lon, R * 1.01);
    const front = v.clone().normalize().dot(camDir);
    tmpV.copy(v).project(camera);
    return { i, el: labelPool[i], front, x: (tmpV.x * 0.5 + 0.5) * w, y: (-tmpV.y * 0.5 + 0.5) * h, z: tmpV.z };
  }).sort((a, b) => b.front - a.front);       // the most face-on cities win the space

  for (const c of cities) {
    const visible = c.front > 0.12 && Math.abs(c.z) < 1;
    if (!visible) { c.el.classList.remove('on'); continue; }
    const box = { x0: c.x - 6, x1: c.x + c.el.textContent.length * 6.2, y0: c.y - 11, y1: c.y + 6 };
    const clash = placed.some((p) => box.x0 < p.x1 && box.x1 > p.x0 && box.y0 < p.y1 && box.y1 > p.y0);
    if (clash) { c.el.classList.remove('on'); continue; }
    placed.push(box);
    c.el.style.left = `${c.x}px`;
    c.el.style.top = `${c.y}px`;
    c.el.style.opacity = String(Math.min(0.95, 0.25 + c.front * 0.8));
    c.el.classList.add('on');
  }
}

// ---------------------------------------------------------------- animation / flight
let flight = null;
function flyTo(lat, lon, distance = R * 1.9, ms = 1100) {
  const from = camera.position.clone();
  const to = toVec(lat, lon).normalize().multiplyScalar(distance);
  flight = { from, to, start: performance.now(), ms };
}

// The hazard wash is thinned as the camera closes in: at 2.9 R the cells read as a global
// pattern, but at 1.9 R a solid 52% red cap would hide the ground beneath it.
const HAZARD_ALPHA_BY_BUCKET = [0.52, 0.4, 0.3];
function syncHazardAlpha() {
  if (!hazardState.on || !hazardState.cells.length) return;
  const d = camera.position.length() / R;
  const bucket = d > 2.4 ? 0 : d > 1.9 ? 1 : 2;
  if (bucket === hazardState.bucket) return;
  hazardState.bucket = bucket;
  hazardState.alpha = HAZARD_ALPHA_BY_BUCKET[bucket];
  applySurface();
}

function stepFlight(now) {
  if (!flight) return;
  const t = Math.min(1, (now - flight.start) / flight.ms);
  const k = t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2;      // ease in-out
  const a = flight.from.clone().normalize();
  const b = flight.to.clone().normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(a, b);
  const qq = new THREE.Quaternion().slerp(q, k);
  const len = THREE.MathUtils.lerp(flight.from.length(), flight.to.length(), k);
  camera.position.copy(a.clone().applyQuaternion(qq).multiplyScalar(len));
  camera.lookAt(0, 0, 0);
  if (t >= 1) flight = null;
}

// ---------------------------------------------------------------- point picking + panel
const app = $('app');
const state = { point: null, controls: null, predictor: null, options: null };

function countryAt(lat, lon) {
  // simple even-odd test over every ring; 177 countries is cheap enough per click
  const inRing = (ring) => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if (((yi > lat) !== (yj > lat)) && (lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  };
  const area = (ring) => {
    let a = 0;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      a += (ring[j][0] * ring[i][1]) - (ring[i][0] * ring[j][1]);
    }
    return Math.abs(a / 2);
  };
  for (const c of countriesData) {
    let inside = false;
    for (const ring of c.rings) {
      if (area(ring) <= 12) continue;            // skip holes and specks for the outer test
      if (inRing(ring)) { inside = true; break; }
    }
    if (inside) return c.name;
  }
  return 'Open sea';
}

function nearestCity(lat, lon) {
  let best = null;
  const cosLat = Math.cos(lat * Math.PI / 180);
  for (const c of citiesData) {
    const dx = (c.lon - lon) * cosLat;
    const dy = c.lat - lat;
    const d = dx * dx + dy * dy;
    if (!best || d < best.d) best = { d, c };
  }
  if (!best) return null;
  return { ...best.c, km: Math.round(Math.sqrt(best.d) * 111) };
}

function setBadge(grade, label) {
  const g = GRADES[grade];
  const badge = $('badge');
  badge.style.background = g.hex;
  $('badgeK').textContent = `Predicted damage · grade ${grade}`;
  $('badgeV').textContent = label || g.label;
}

function selectPoint(lat, lon, openSheet = true) {
  const marker = globe.userData.marker;
  if (marker) globe.remove(marker);
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.85, 18, 14),
    new THREE.MeshBasicMaterial({ color: '#ffe6ad' }),
  );
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(1.5, 2.3, 32),
    new THREE.MeshBasicMaterial({ color: '#ffd98a', transparent: true, opacity: 0.75, side: THREE.DoubleSide }),
  );
  m.position.copy(toVec(lat, lon, R * 1.012));
  // the ring sits flat on the surface, just under the dot
  ring.position.copy(m.position.clone().normalize().multiplyScalar(-0.35));
  ring.lookAt(m.position.clone().multiplyScalar(2));
  m.add(ring);
  globe.add(m);
  globe.userData.marker = m;

  state.point = { lat, lon };
  $('pLat').textContent = `${lat.toFixed(4)}°`;
  $('pLon').textContent = `${lon.toFixed(4)}°`;
  const country = countryAt(lat, lon);
  $('pCountry').textContent = country;
  const near = nearestCity(lat, lon);
  $('pNear').textContent = near ? `${near.name} · ${fmt(near.km)} km` : '–';
  $('placeName').textContent = near && near.km < 400 ? `${near.name}` : country;
  $('hint').textContent = country === 'Open sea'
    ? 'You picked a point at sea — the prediction still works, because the model reads the building profile below rather than the coordinates.'
    : 'Coordinates are shown for reference only. The prediction uses the building profile below, because the training survey has no coordinates.';

  if (openSheet) {
    app.dataset.sheetOpen = '1';
    app.classList.add('open');
    window.dispatchEvent(new Event('globe:sheet'));   // the example panel steps aside
  }
  if (state.predictor && state.controls) state.predictor.now(state.controls.read());
}

function pickAt(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(globe.userData.pickTarget, false)[0];
  if (!hit) return;
  const local = globe.userData.pickTarget.worldToLocal(hit.point.clone());
  const [lat, lon] = toLatLon(local);
  selectPoint(lat, lon);
}

const closeSheet = () => {
  app.dataset.sheetOpen = '0';
  app.classList.remove('open');
  window.dispatchEvent(new Event('globe:sheet'));
};
$('closeSheet').addEventListener('click', closeSheet);
// the example panel asks for the column: give it up and let it draw
window.addEventListener('globe:examples-open', closeSheet);
// without WebGL there is no canvas to listen on, so the pick handlers are skipped
if (renderer) renderer.domElement.addEventListener('pointerdown', (ev) => { renderer.domElement.dataset.down = `${ev.clientX},${ev.clientY}`; });
if (renderer) renderer.domElement.addEventListener('pointerup', (ev) => {
  const down = renderer.domElement.dataset.down;
  if (!down) return;
  const [dx, dy] = down.split(',').map(Number);
  if (Math.hypot(ev.clientX - dx, ev.clientY - dy) < 6) pickAt(ev);   // a click, not a drag
});

// ---------------------------------------------------------------- prediction form
async function initForm() {
  const options = await loadOptions();
  state.options = options;
  state.predictor = createPredictor({
    onLoading: (on) => {
      let spin = document.querySelector('#sheet .spin');
      if (!spin) {
        spin = document.createElement('span');
        spin.className = 'spin';
        $('badgeK').appendChild(spin);
      }
      spin.classList.toggle('on', on);
    },
    onResult: (out) => {
      setBadge(out.damage_grade, out.label);
      renderProbBars($('probs'), out.probabilities);
    },
    onError: (msg) => {
      $('badgeK').textContent = 'Prediction failed';
      $('badgeV').textContent = msg;
    },
  });
  state.controls = buildControls($('form'), options, (payload) => state.predictor.schedule(payload));
  state.predictor.now(state.controls.read());
}

// ---------------------------------------------------------------- search
let searchIndex = [];
function initSearch() {
  const input = $('q');
  const box = $('results');

  const render = (rows) => {
    if (!rows.length) { box.style.display = 'none'; return; }
    box.innerHTML = rows.map((r, i) => `<button data-i="${i}"><b>${r.name}</b><small>${r.kind}${r.sub ? ' · ' + r.sub : ''}</small></button>`).join('');
    box.style.display = 'block';
    box.querySelectorAll('button').forEach((b) => {
      b.addEventListener('click', () => {
        const r = rows[Number(b.dataset.i)];
        flyTo(r.lat, r.lon, R * 1.75, 1000);
        selectPoint(r.lat, r.lon, r.kind === 'City');
        input.value = r.name;
        box.style.display = 'none';
      });
    });
  };

  input.addEventListener('input', () => {
    const s = input.value.trim().toLowerCase();
    if (s.length < 2) { box.style.display = 'none'; return; }
    const rows = searchIndex
      .filter((r) => r.key.startsWith(s) || r.key.includes(' ' + s))
      .slice(0, 8);
    render(rows);
  });
  input.addEventListener('keydown', (e) => { if (e.key === 'Escape') { box.style.display = 'none'; input.blur(); } });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#search')) box.style.display = 'none';
  });
}

function buildSearchIndex(countries, cities) {
  const out = countries.map((c) => {
    const ring = c.rings[0];
    const lat = ring.reduce((a, p) => a + p[1], 0) / ring.length;
    const lon = ring.reduce((a, p) => a + p[0], 0) / ring.length;
    return { kind: 'Country', name: c.name, key: c.name.toLowerCase(), lat, lon, sub: '' };
  });
  for (const c of cities.slice(0, 400)) {
    out.push({ kind: 'City', name: c.name, key: c.name.toLowerCase(), lat: c.lat, lon: c.lon, sub: `${c.country} · ${fmt(c.pop)}` });
  }
  searchIndex = out;
}

// ---------------------------------------------------------------- boot
function setLoad(pct, text) {
  const bar = $('loadbar'), label = $('loadtext');
  if (bar) bar.style.width = `${pct}%`;
  if (label) label.textContent = text;      // the no-WebGL message replaces the loader card
}

async function boot() {
  fit();
  DOT_TEX = dotTexture();
  setLoad(8, 'Reading the country geometry…');
  const [countries, cities, status] = await Promise.all([
    fetch('/api/world/countries').then((r) => r.json()),
    fetch('/api/world/cities?min_pop=100000&limit=3000').then((r) => r.json()),
    fetch('/api/world/status').then((r) => r.json()),
  ]);
  countriesData = countries.countries;
  citiesData = cities.cities;

  setLoad(34, `Building the globe from ${countries.stats.countries} countries…`);
  if (!window.__noGL) buildGlobe(countriesData);
  await sleep(30);

  setLoad(58, 'Loading the earthquake catalogue…');
  const events = await fetch('/api/world/events?min_mag=5.0&limit=30000').then((r) => r.json());
  if (!window.__noGL) buildEvents(events.events);

  setLoad(78, 'Scoring the hazard grid…');
  const hazard = await fetch('/api/world/hazard').then((r) => r.json());
  if (!window.__noGL) buildHazard(hazard.cells, hazard.meta);

  setLoad(92, 'Adding cities and labels…');
  if (!window.__noGL) buildLabels(citiesData);
  buildSearchIndex(countriesData, citiesData);

  $('st-countries').textContent = fmt(countries.stats.countries);
  $('st-events').textContent = fmt(events.count);
  $('st-strong').textContent = fmt(events.events.filter((e) => e.mag >= 7).length);
  $('st-cells').textContent = fmt(hazard.meta.cells_total ?? hazard.cells.length);
  $('srcLine').textContent = `${events.source}. Hazard cells from the same catalogue (${hazard.meta.cell_deg}° cells). Countries: Natural Earth 1:110m. Cities: GeoNames.`;
  $('status').textContent = `${events.source.split(',')[0]} · ${fmt(events.count)} events · ${fmt(hazard.cells.length)} hazard cells`;

  await initForm();
  if (!window.__noGL) {
    setLoad(100, 'Ready');
    $('loader').classList.add('done');
  }

  if (status.catalogue && status.catalogue.warming) {
    setTimeout(async () => {
      // pick up the live USGS catalogue once the background fetch finishes
      try {
        const fresh = await fetch('/api/world/events?min_mag=5.0&limit=30000').then((r) => r.json());
        if (fresh.coverage === 'global' && !fresh.warming) {
          buildEvents(fresh.events);
          const hz = await fetch('/api/world/hazard?refresh=true').then((r) => r.json());
          buildHazard(hz.cells, hz.meta);
          $('status').textContent = `${fresh.source} · ${fmt(fresh.count)} events`;
        }
      } catch (_) { /* keep the bundled catalogue */ }
    }, 12000);
  }
}

function loop(now) {
  if (!renderer) return;
  stepFlight(now);
  syncHazardAlpha();
  if (controls) controls.update();
  updateLabels();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

document.querySelectorAll('#dock button').forEach((b) => {
  b.addEventListener('click', () => {
    const lat = Number(b.dataset.lat), lon = Number(b.dataset.lon);
    flyTo(lat, lon, R * 1.75, 1200);
    selectPoint(lat, lon);
  });
});

// layer toggles
$('tglHazard').addEventListener('click', (e) => {
  hazardState.on = !hazardState.on;
  e.currentTarget.classList.toggle('on', hazardState.on);
  e.currentTarget.setAttribute('aria-pressed', String(hazardState.on));
  applySurface();
});
$('tglEvents').addEventListener('click', (e) => {
  const on = !(eventPoints[0] && eventPoints[0].visible);
  eventPoints.forEach((p) => { p.visible = on; });
  e.currentTarget.classList.toggle('on', on);
  e.currentTarget.setAttribute('aria-pressed', String(on));
});

// handy for debugging and for the browser tests
window.__globe = {
  scene, camera, globe,
  get layers() { return { events: eventPoints, hazard: hazardState }; },
  toggleHazard(v) { hazardState.on = v; applySurface(); },
  toggleEvents(v) { eventPoints.forEach((p) => { p.visible = v; }); },
};

fit();
requestAnimationFrame(loop);
boot().catch((err) => {
  $('loadtext').textContent = `Could not start the globe: ${err.message}`;
  $('status').textContent = err.message;
});
