// The 3D "places" view: Indian regions hit by large earthquakes, drawn from the real
// USGS catalogue. Assam keeps its own page (13.3 million Overture footprints); the
// places here have no bundled footprints, so the scene shows what the catalogue does
// support — a smoothed hazard grid as extruded cells, and the epicentres themselves.
import * as THREE from 'three';
import { OrbitControls } from '/static/assam3d/vendor/OrbitControls.js';
import { GRADES, loadOptions, buildControls, createPredictor, renderProbBars, SHARED_CSS } from '/static/shared/shared.js';

const $ = (id) => document.getElementById(id);
document.getElementById('shared-css').textContent = SHARED_CSS;

const HAZARD_HEX = { r: '#c53030', y: '#d69e2e', g: '#2f855a' };
const MAG_HEX = (m) => (m >= 7 ? '#ff6b4a' : m >= 6 ? '#f0a43a' : '#f6d28a');

const placeId = location.pathname.split('/').pop() || 'kutch';
const R = 100;                                  // scene radius, same convention as the globe
const state = { place: null, cells: [], events: [] };

// ------------------------------------------------------------------ scene
const container = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#0a1322');
scene.fog = new THREE.Fog('#0a1322', R * 3.2, R * 7);

const camera = new THREE.PerspectiveCamera(40, 1, 0.5, R * 40);
camera.position.set(0, R * 1.2, R * 1.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.7;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = R * 0.6;
controls.maxDistance = R * 4.2;

scene.add(new THREE.HemisphereLight('#cfe0f2', '#0b1626', 1.05));
const sun = new THREE.DirectionalLight('#fff6e5', 1.5);
sun.position.set(-R, R * 2.2, R * 0.8);
scene.add(sun);

function fit() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// Pull the camera back until the whole place box is in frame on both axes.
function frameBox() {
  const wdeg = box.w * R / 1.6, hdeg = box.h * R / 1.6;
  const tan = Math.tan((camera.fov * Math.PI) / 360);
  const fitH = hdeg / (2 * tan) * 1.15;
  const fitW = wdeg / (2 * tan) / Math.max(0.55, camera.aspect) * 1.15;
  const d = Math.max(fitH, fitW, R * 0.9);
  camera.position.set(0, d * 0.52, d * 0.86);
  controls.target.set(0, 1.5, 0);
  controls.maxDistance = d * 2.6;
  camera.lookAt(0, 1.5, 0);
  controls.update();
}
window.addEventListener('resize', fit);

// ------------------------------------------------------------------ geometry helpers
// The place box is mapped onto a plane: lon -> x, lat -> -z. One scene unit per
// degree keeps the aspect right at these latitudes without a projection library.
let box = { west: 0, south: 0, east: 0, north: 0, cx: 0, cz: 0, w: 0, h: 0 };

function toWorld(lon, lat) {
  return { x: (lon - box.cx) * R / 1.6, z: -(lat - box.cz) * R / 1.6 };
}

function buildGround() {
  const w = box.w * R / 1.6, h = box.h * R / 1.6;
  const group = new THREE.Group();

  // a dark plate with a soft edge, so the box reads as a cut-out of the region
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(w * 1.06, h * 1.06),
    new THREE.MeshStandardMaterial({ color: '#132a44', roughness: 0.95, metalness: 0.05 }),
  );
  plate.rotation.x = -Math.PI / 2;
  plate.position.y = -0.15;
  group.add(plate);

  // graticule every degree
  const pts = [];
  for (let lon = Math.ceil(box.west); lon <= box.east; lon += 1) {
    const a = toWorld(lon, box.south), b = toWorld(lon, box.north);
    pts.push(a.x, 0, a.z, b.x, 0, b.z);
  }
  for (let lat = Math.ceil(box.south); lat <= box.north; lat += 1) {
    const a = toWorld(box.west, lat), b = toWorld(box.east, lat);
    pts.push(a.x, 0, a.z, b.x, 0, b.z);
  }
  const gridGeo = new THREE.BufferGeometry();
  gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  group.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({
    color: '#4d6a8c', transparent: true, opacity: 0.4,
  })));

  scene.add(group);
  return group;
}

function buildCells(cells, cellDeg) {
  const geo = new THREE.BoxGeometry(1, 1, 1);
  geo.translate(0, 0.5, 0);
  const scores = cells.map((c) => c[3]);
  const lo = Math.min(...scores), hi = Math.max(...scores) || 1;
  const side = cellDeg * R / 1.6;                 // exactly one cell wide, so the grid tiles

  const mesh = new THREE.InstancedMesh(
    geo,
    new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.1, vertexColors: false }),
    cells.length,
  );
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cells.length * 3), 3);

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const col = new THREE.Color();
  cells.forEach(([lon, lat, cls, score], i) => {
    const { x, z } = toWorld(lon + cellDeg / 2, lat + cellDeg / 2);
    const t = hi > lo ? (score - lo) / (hi - lo) : 0.5;
    const hgt = 0.6 + t * t * 9;                     // a few tall columns, not a wall
    m.compose(new THREE.Vector3(x, 0, z), q, new THREE.Vector3(side, hgt, side));
    mesh.setMatrixAt(i, m);
    col.set(HAZARD_HEX[cls] || '#888');
    mesh.setColorAt(i, col);
  });
  mesh.instanceMatrix.needsUpdate = true;
  scene.add(mesh);
  return mesh;
}

const TOP = 11;             // epicentres float just above the tallest column
const pinNodes = [];        // the three biggest events get a floating label

function buildEvents(events) {
  const group = new THREE.Group();
  const dot = new THREE.SphereGeometry(1, 12, 10);
  const lineMat = new THREE.LineBasicMaterial({ color: '#ffd98a', transparent: true, opacity: 0.55 });

  events.forEach((e) => {
    const { x, z } = toWorld(e.lon, e.lat);
    const size = 0.45 + Math.max(0, e.mag - 4) * 0.55;
    const y = TOP + size;
    const m = new THREE.Mesh(dot, new THREE.MeshBasicMaterial({
      color: MAG_HEX(e.mag), transparent: true, opacity: 0.95,
    }));
    m.position.set(x, y, z);
    m.scale.setScalar(size);
    group.add(m);

    // a thin stem ties the epicentre to its point on the ground
    const stem = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x, 0, z), new THREE.Vector3(x, y, z),
    ]);
    group.add(new THREE.Line(stem, lineMat));

    if (e.mag >= 6) {                          // ring the ones that did real damage
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(size * 1.7, size * 2.1, 28),
        new THREE.MeshBasicMaterial({ color: MAG_HEX(e.mag), transparent: true, opacity: 0.6, side: THREE.DoubleSide }),
      );
      ring.position.set(x, y, z);
      ring.rotation.x = -Math.PI / 2;
      group.add(ring);
    }
  });
  // labels for the largest events: rebuilt whenever the place changes
  pinNodes.length = 0;
  $('pins').innerHTML = '';
  // label the largest events, but only ones that are far enough apart to read
  const biggest = [];
  for (const e of [...events].sort((a, b) => b.mag - a.mag)) {
    if (biggest.length === 3) break;
    const far = biggest.every((o) => Math.hypot(o.lon - e.lon, o.lat - e.lat) > 1.1);
    if (far) biggest.push(e);
  }
  const lift = [3.2, 4.6, 6.0];
  biggest.forEach((e, i) => {
    const el = document.createElement('div');
    el.className = 'pin';
    el.innerHTML = `<b>M${e.mag.toFixed(1)}</b> ${e.place.split(',')[0]} · ${e.time.slice(0, 4)}`;
    el.style.zIndex = String(10 - i);
    $('pins').appendChild(el);
    const { x, z } = toWorld(e.lon, e.lat);
    pinNodes.push({ el, pos: new THREE.Vector3(x, TOP + lift[i], z) });
  });

  scene.add(group);
  return group;
}

const _v = new THREE.Vector3();
const panels = ['info', 'legend'].map((id) => {
  const el = $(id);
  return () => {
    const r = el.getBoundingClientRect();
    const s = container.getBoundingClientRect();
    return { x0: r.left - s.left, y0: r.top - s.top, x1: r.right - s.left, y1: r.bottom - s.top };
  };
});

function updatePins() {
  if (!pinNodes.length) return;
  const w = container.clientWidth, h = container.clientHeight;
  const boxes = panels.map((f) => f());
  for (const pin of pinNodes) {
    _v.copy(pin.pos).project(camera);
    const onStage = _v.z < 1 && Math.abs(_v.x) < 0.98 && Math.abs(_v.y) < 0.98
      && (_v.x * 0.5 + 0.5) * w > 70 && (-_v.y * 0.5 + 0.5) * h > 40;
    const x = (_v.x * 0.5 + 0.5) * w;
    const y = (-_v.y * 0.5 + 0.5) * h;
    // a label that lands on a card or off the stage is worse than no label
    const hidden = boxes.some((b) => x > b.x0 - 60 && x < b.x1 + 20 && y > b.y0 - 20 && y < b.y1 + 20);
    pin.el.style.display = onStage && !hidden ? 'block' : 'none';
    if (!onStage) continue;
    pin.el.style.left = `${x}px`;
    pin.el.style.top = `${y}px`;
  }
}

let disposed = [];

function clearScene() {
  for (const obj of disposed) {
    scene.remove(obj);
    obj.traverse?.((o) => {
      o.geometry?.dispose?.();
      if (Array.isArray(o.material)) o.material.forEach((mm) => mm.dispose());
      else o.material?.dispose?.();
    });
  }
  disposed = [];
}

// ------------------------------------------------------------------ place data
function setStatus(text) { $('status').textContent = text; }

function fmt(n) { return Number(n).toLocaleString('en-US'); }

async function boot() {
  fit();
  let data;
  try {
    const r = await fetch(`/api/places/${encodeURIComponent(placeId)}`);
    if (!r.ok) throw new Error(`server returned ${r.status}`);
    data = await r.json();
  } catch (err) {
    $('loadTitle').textContent = 'Could not load this place';
    $('loadText').textContent = String(err.message);
    return;
  }
  state.place = data;
  state.cells = data.cells || [];
  state.events = data.events || [];

  document.title = `${data.name} · earthquake hazard in 3D`;
  $('placeName').textContent = data.name;
  $('sideName').textContent = data.name;
  $('placeBlurb').textContent = data.headline;
  $('navAssam').classList.toggle('on', data.id === 'assam');
  $('navKutch').classList.toggle('on', data.id === 'kutch');
  $('navKoyna').classList.toggle('on', data.id === 'koyna');

  const meta = data.meta;
  const [west, south, east, north] = data.bbox;
  box = { west, south, east, north, cx: (west + east) / 2, cz: (south + north) / 2, w: east - west, h: north - south };

  $('loadTitle').textContent = data.name;
  $('loadText').textContent = 'Drawing the hazard grid…';

  frameBox();
  disposed.push(buildGround());
  disposed.push(buildCells(state.cells, meta.cell_deg));
  disposed.push(buildEvents(state.events));

  $('stEvents').textContent = fmt(meta.events);
  $('stM5').textContent = fmt(meta.events_m5);
  $('stCells').textContent = fmt(meta.cells_total);
  $('stMax').textContent = meta.largest ? `M${meta.largest.mag.toFixed(1)}` : '–';
  $('methodLine').textContent = meta.method;
  $('sourceLine').textContent = `${meta.source}. Cell size ${meta.cell_deg}°, smoothing σ ${meta.smoothing_sigma_deg}°.`;
  setStatus(`${meta.source} · ${fmt(meta.events)} events · ${fmt(meta.cells_total)} hazard cells`);

  const big = [...state.events].sort((a, b) => b.mag - a.mag).slice(0, 6);
  $('bigList').innerHTML = big.map((e) => (
    `<div><b>M${e.mag.toFixed(1)}</b><span>${e.time} · ${e.place || `${e.lat.toFixed(2)}, ${e.lon.toFixed(2)}`}</span></div>`
  )).join('') || '<div><span>No events in this box.</span></div>';

  $('loader').classList.add('done');
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updatePins();
  renderer.render(scene, camera);
}

// ------------------------------------------------------------------ damage model
async function initModel() {
  const options = await loadOptions();
  const predictor = createPredictor({
    onResult: (d) => {
      const g = d.damage_grade;
      const grade = GRADES[g];
      $('badge').style.background = grade.hex;
      $('badgeK').textContent = `Predicted damage · grade ${g}`;
      $('badgeV').textContent = grade.label;
      renderProbBars($('probs'), d.probabilities, GRADES);
    },
    onLoading: (busy) => { if (busy) $('badgeV').textContent = 'Calculating…'; },
    onError: (msg) => { $('badgeV').textContent = 'Could not predict: ' + msg; },
  });
  buildControls($('form'), options, (payload) => predictor.schedule(payload));
  // one immediate run, so the panel is never an empty dash
  $('form').dispatchEvent(new Event('change'));
}

boot().then(initModel).catch((err) => {
  $('loadTitle').textContent = 'Something went wrong';
  $('loadText').textContent = String(err.message || err);
});
