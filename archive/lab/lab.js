// Building lab: a 3D building that responds to the damage model in real time.
import * as THREE from 'three';
import { OrbitControls } from '/static/assam3d/vendor/OrbitControls.js';
import { GRADES, SHARED_CSS, loadOptions, buildControls, createPredictor, renderProbBars } from '/static/shared/shared.js';

document.getElementById('shared-css').textContent = SHARED_CSS;

const $ = (id) => document.getElementById(id);

// Material colours, strongest first. The first ticked material sets the wall colour.
const MATERIALS = [
  ['has_superstructure_rc_engineered', 0x9ea6b0, 'engineered reinforced concrete'],
  ['has_superstructure_rc_non_engineered', 0xb5b0a6, 'non-engineered reinforced concrete'],
  ['has_superstructure_cement_mortar_brick', 0xb5573f, 'brick in cement mortar'],
  ['has_superstructure_mud_mortar_brick', 0xc7825a, 'brick in mud mortar'],
  ['has_superstructure_cement_mortar_stone', 0x8f887c, 'stone in cement mortar'],
  ['has_superstructure_stone_flag', 0x7f7a70, 'stone flag'],
  ['has_superstructure_mud_mortar_stone', 0xa88c67, 'stone in mud mortar'],
  ['has_superstructure_timber', 0x8b6a43, 'timber'],
  ['has_superstructure_bamboo', 0xc9b46a, 'bamboo'],
  ['has_superstructure_adobe_mud', 0xcba877, 'adobe / mud'],
];

// ------------------------------------------------------------------ renderer
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
// soft sky gradient as the background
{
  const c = document.createElement('canvas'); c.width = 4; c.height = 256;
  const g = c.getContext('2d'); const grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, '#cfe1f1'); grd.addColorStop(0.6, '#eef3f7'); grd.addColorStop(1, '#e8e4da');
  g.fillStyle = grd; g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  scene.background = tex;
}

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 1000);
const HOME = new THREE.Vector3(30, 22, 34);
camera.position.copy(HOME);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 7, 0);
controls.enableDamping = true;
controls.minDistance = 12;
controls.maxDistance = 120;
controls.maxPolarAngle = Math.PI * 0.49;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

scene.add(new THREE.HemisphereLight('#ffffff', '#b8b09f', 1.25));
const sun = new THREE.DirectionalLight('#fff6e8', 2.1);
sun.position.set(18, 34, 22);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -40, right: 40, top: 40, bottom: -40, near: 1, far: 120 });
sun.shadow.bias = -0.0006;
scene.add(sun);

// ground: a shadow-receiving plane, with a faint survey grid
const ground = new THREE.Mesh(new THREE.CircleGeometry(160, 96),
  new THREE.MeshStandardMaterial({ color: '#e7e1d3', roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(120, 24, '#c9c2b2', '#d8d2c4');
grid.position.y = 0.01;
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);

// ------------------------------------------------------------------ building
const building = new THREE.Group();
scene.add(building);

const wallMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.82, metalness: 0 });
const body = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), wallMat);
body.position.y = 0.5;
body.castShadow = true; body.receiveShadow = true;
building.add(body);

const roof = new THREE.Mesh(new THREE.BoxGeometry(1, 0.35, 1), new THREE.MeshStandardMaterial({ color: '#4a5260', roughness: 0.7 }));
roof.castShadow = true;
building.add(roof);

const plate = new THREE.Mesh(new THREE.BoxGeometry(1, 0.3, 1), new THREE.MeshStandardMaterial({ color: '#7a8699', roughness: 0.5, emissive: '#000000', emissiveIntensity: 0.25 }));
plate.receiveShadow = true;
building.add(plate);

// current and target shape (lerped each frame for smooth changes)
const cur = { w: 10, d: 8, h: 6, plate: 0x7a8699 };
const target = { w: 10, d: 8, h: 6 };
const plateColor = new THREE.Color(0x7a8699);
let plateTarget = new THREE.Color(0x7a8699);
let wallColor = new THREE.Color(0xc8c2b4);
let wallTarget = new THREE.Color(0xc8c2b4);

function windowTexture(floors, base) {
  const W = 512, H = 512, cols = 4, rows = Math.max(1, floors);
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = '#' + base.getHexString(); g.fillRect(0, 0, W, H);
  const cw = W / cols, rh = H / rows;
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < cols; k++) {
      const x = k * cw + cw * 0.22, y = r * rh + rh * 0.2;
      g.fillStyle = 'rgba(40,58,78,0.78)';
      g.fillRect(x, y, cw * 0.56, rh * 0.52);
      g.fillStyle = 'rgba(255,255,255,0.25)';
      g.fillRect(x, y, cw * 0.56, rh * 0.09);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

let texFloors = -1, texBase = null;
function applyWallTexture(floors) {
  if (floors === texFloors && texBase && texBase.equals(wallTarget)) return;
  if (wallMat.map) wallMat.map.dispose();
  wallMat.map = windowTexture(floors, wallTarget.clone());
  wallMat.needsUpdate = true;
  texFloors = floors;
  texBase = wallTarget.clone();
}

// ------------------------------------------------------------------ state from the form
let lastPayload = null;
let currentGrade = null;
let grades = { 1: 0, 2: 0, 3: 0 };

function visualFromPayload(p) {
  const floors = Number(p.count_floors_pre_eq) || 1;
  const plinth = Number(p.area_percentage) || 1;
  target.h = floors * 3.0;
  target.w = 7 + Math.min(plinth, 100) * 0.14;
  target.d = target.w * 0.78;

  let colour = 0xa9b4c0, label = 'no structural material ticked';
  for (const [name, hex, text] of MATERIALS) {
    if (Number(p[name]) === 1) { colour = hex; label = text; break; }
  }
  wallTarget = new THREE.Color(colour);
  applyWallTexture(floors);
  return { floors, label };
}

const predictor = createPredictor({
  onLoading: (on) => $('spin').classList.toggle('on', on),
  onResult: (out) => showResult(out),
  onError: (msg) => { $('badge-v').textContent = 'Could not predict: ' + msg; },
});

function showResult(out) {
  const g = Number(out.damage_grade);
  currentGrade = g;
  const meta = GRADES[g];
  $('badge').style.background = meta.hex;
  $('badge-v').textContent = `${g} · ${meta.label}`;
  renderProbBars($('bars'), out.probabilities);
  plateTarget = new THREE.Color(meta.hex);
  $('modelline').textContent = `Model: ${out.model} · ${out.disclaimer.split('.')[0]}.`;
  const fl = lastPayload ? lastPayload.count_floors_pre_eq : '';
  const mat = lastPayload ? visualFromPayload(lastPayload).label : '';
  $('why').innerHTML = `Inputs: ${fl} floors, wall material ${mat}. Colour key: <span class="swatch" style="background:${GRADES[1].hex}"></span>low <span class="swatch" style="background:${GRADES[2].hex}"></span>medium <span class="swatch" style="background:${GRADES[3].hex}"></span>near-complete.`;
}

// ------------------------------------------------------------------ controls
loadOptions().then((options) => {
  buildControls($('controls'), options, (payload) => {
    lastPayload = payload;
    visualFromPayload(payload);
    predictor.schedule(payload);
  });
  const init = Object.fromEntries(options.fields.map((f) => [f.name, f.default]));
  // the form starts from the defaults; set the initial shape from them
  lastPayload = init;
  visualFromPayload(init);
  predictor.now(init);
}).catch((e) => {
  $('badge-v').textContent = 'Options unavailable: ' + e.message;
});

// ------------------------------------------------------------------ buttons
$('rotate').addEventListener('click', () => {
  controls.autoRotate = !controls.autoRotate;
  $('rotate').classList.toggle('on', controls.autoRotate);
});
$('reset').addEventListener('click', () => {
  camera.position.copy(HOME);
  controls.target.set(0, 7, 0);
  controls.update();
});

// ------------------------------------------------------------------ loop
function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px';
  renderer.domElement.style.height = h + 'px';
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

const ease = 0.12;
renderer.setAnimationLoop(() => {
  cur.w += (target.w - cur.w) * ease;
  cur.d += (target.d - cur.d) * ease;
  cur.h += (target.h - cur.h) * ease;
  body.scale.set(cur.w, cur.h, cur.d);
  body.position.y = cur.h / 2;
  roof.scale.set(cur.w + 0.6, 1, cur.d + 0.6);
  roof.position.y = cur.h + 0.175;
  plate.scale.set(cur.w + 2.4, 1, cur.d + 2.4);
  plate.position.y = 0.15;
  plateColor.lerp(plateTarget, 0.08);
  plate.material.color.copy(plateColor);
  plate.material.emissive.copy(plateColor);
  controls.update();
  renderer.render(scene, camera);
});
