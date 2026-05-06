// Unlicense — cochranblock.org

// ── Palette (endesga32 — site default; pico-8 reserved for phone app) ───
const PALETTES = {
  endesga32: ['#be4a2f','#d77643','#ead4aa','#e4a672','#b86f50','#733e39','#3e2731','#a22633','#e43b44','#f77622','#feae34','#fee761','#63c74d','#3e8948','#265c42','#193c3e','#124e89','#0099db','#2ce8f5','#ffffff','#c0cbdc','#8b9bb4','#5a6988','#3a4466','#262b44','#181425','#ff0044','#68386c','#b55088','#f6757a','#e8b796','#c28569'],
};

const OUTLINE = '#181425'; // endesga32 near-black — consistent outline across all sprites

// 2×2 Bayer matrix for ordered dithering (crisp shading bands, no random noise)
const BAYER = [[0, 2], [3, 1]];

function brightness(hex) {
  const r=parseInt(hex.slice(1,3),16), g=parseInt(hex.slice(3,5),16), b=parseInt(hex.slice(5,7),16);
  return 0.299*r + 0.587*g + 0.114*b;
}

// ── PRNG (mulberry32 variant) ─────────────
function mkRng(seed) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  return function () {
    h |= 0; h = h + 0x6D2B79F5 | 0;
    let t = Math.imul(h ^ h >>> 15, 1 | h);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ── Class-specific silhouette density ────────
// nx: 0=outer edge → 1=center (left half, mirrored)
// ny: 0=top → 1=bottom
// Returns fill probability; ≤0 means skip.
function shapeDensity(cls, nx, ny) {
  const e = 1 - nx; // e=0 at center, e=1 at outer edge
  switch (cls) {
    case 'character': {
      // Head: oval at top-center
      if (ny < 0.42) {
        const dx = e / 0.34, dy = (ny - 0.21) / 0.21;
        return 1.1 - dx * dx - dy * dy;
      }
      // Torso
      if (ny < 0.70) return e < 0.54 ? 0.9 : -1;
      // Legs: two strips (gap at center so they read as separate)
      return (e > 0.17 && e < 0.47) ? 0.88 : -1;
    }
    case 'dragon': {
      // Wide oval body, heavier toward top (big head)
      const body = 0.9 - (e / 0.78) * (e / 0.78) - ((ny - 0.45) / 0.42) * ((ny - 0.45) / 0.42);
      // Head bump: fills inner half at very top
      const head = (ny < 0.30 && e < 0.50) ? 0.8 - e * 0.8 - ny * 1.8 : -1;
      return Math.max(body, head);
    }
    case 'sword': {
      if (e < 0.15 && ny < 0.78) return 0.95;           // blade
      if (ny > 0.72 && ny < 0.84 && e < 0.72) return 0.92; // crossguard
      if (e < 0.15 && ny >= 0.84) return 0.88;           // handle
      return -1;
    }
    case 'potion': {
      // Round body
      if (ny > 0.30) {
        const dx = e / 0.56, dy = (ny - 0.70) / 0.31;
        return 1.1 - dx * dx - dy * dy;
      }
      // Narrow neck
      return e < 0.20 ? 0.9 : -1;
    }
    case 'tree': {
      // Triangular canopy: maxEdge grows with ny
      if (ny < 0.70) return e < ny * 0.78 + 0.06 ? 0.9 : -1;
      // Thin trunk
      return e < 0.20 ? 0.85 : -1;
    }
    case 'building': {
      if (e > 0.82) return -1;
      // Window cutout (low density = mostly dark gap)
      if (ny > 0.28 && ny < 0.58 && e > 0.12 && e < 0.50) return 0.08;
      return 0.92;
    }
    case 'vehicle': {
      if (ny > 0.30 && ny < 0.70 && e < 0.88) return 0.92; // body
      if (ny <= 0.30 && ny > 0.04 && e < 0.56) return 0.88; // cab
      // Wheel: circle near outer-bottom
      const dx = (e - 0.60) / 0.18, dy = (ny - 0.80) / 0.18;
      return ny > 0.62 ? 0.95 - dx * dx * 3 - dy * dy * 3 : -1;
    }
    case 'furniture': {
      if (ny < 0.18 && e < 0.88) return 0.92;              // tabletop
      if (ny > 0.75 && e > 0.52 && e < 0.76) return 0.88;  // leg
      return -1;
    }
    default: {
      // Generic oval blob
      const dx = e / 0.60, dy = (ny - 0.5) / 0.45;
      return 0.9 - dx * dx - dy * dy;
    }
  }
}

// ── Procedural 16×16 sprite ───────────────
function drawSprite(canvas, cls, palName, seed) {
  const pal = PALETTES[palName] || PALETTES.stardew;
  const rng = mkRng(cls + seed);
  const S = 16;
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext('2d');
  const px = new Uint8Array(S * S);
  const half = Math.ceil(S / 2);

  // Zone-pick 3 colors: one from each brightness third — guaranteed contrast
  const byBright = [...pal].sort((a, b) => brightness(a) - brightness(b));
  const z = Math.floor(byBright.length / 3);
  const dark  = byBright[Math.floor(rng() * z)];
  const mid   = byBright[z   + Math.floor(rng() * z)];
  const light = byBright[2*z + Math.floor(rng() * (byBright.length - 2*z))];
  const colors = [null, dark, mid, light]; // 1=dark 2=mid 3=light

  // Fill left half — shape mask only (no color yet)
  for (let y = 0; y < S; y++)
    for (let x = 0; x < half; x++) {
      const d = shapeDensity(cls, x / half, y / S);
      if (d > 0 && rng() <= d) px[y * S + x] = 1;
    }

  // Mirror left→right
  for (let y = 0; y < S; y++)
    for (let x = 0; x < half; x++)
      px[y * S + (S - 1 - x)] = px[y * S + x];

  // Bayer 2×2 ordered dithering — assign dark/mid/light per pixel
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      if (!px[y * S + x]) continue;
      const shade = 1 - y / S;
      const bayer = BAYER[y % 2][x % 2] / 4.0;
      const d = shade + (bayer - 0.5) * 0.35;
      px[y * S + x] = d > 0.65 ? 3 : d > 0.35 ? 2 : 1;
    }

  // Rim light — top silhouette edge → lightest color
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      if (px[y * S + x] && (y === 0 || !px[(y - 1) * S + x]))
        px[y * S + x] = 3;

  // Outline pass — empty pixels touching a filled cardinal neighbor
  const outline = new Uint8Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (px[y*S+x]) continue;
      if ((y > 0   && px[(y-1)*S+x]) || (y < S-1 && px[(y+1)*S+x]) ||
          (x > 0   && px[y*S+(x-1)]) || (x < S-1 && px[y*S+(x+1)])) {
        outline[y*S+x] = 1;
      }
    }
  }

  // Render: outline then shaded fill
  ctx.clearRect(0, 0, S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      if (outline[y*S+x]) { ctx.fillStyle = OUTLINE; ctx.fillRect(x, y, 1, 1); continue; }
      const c = px[y*S+x];
      if (!c) continue;
      ctx.fillStyle = colors[c];
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

// ── Generate section ──────────────────────
function initGenerate() {
  const clsEl  = document.getElementById('gen-class');
  const cntEl  = document.getElementById('gen-count');
  const cntOut = document.getElementById('gen-count-out');
  const goBtn  = document.getElementById('gen-go');
  const status = document.getElementById('gen-status');
  const grid   = document.getElementById('gen-grid');

  cntEl.addEventListener('input', () => { cntOut.value = cntEl.value; });

  goBtn.addEventListener('click', () => {
    const cls = clsEl.value;
    const pal = 'endesga32';
    const cnt = parseInt(cntEl.value, 10);
    status.textContent = `Generating ${cnt} ${cls}…`;
    grid.innerHTML = '';
    goBtn.disabled = true;

    requestAnimationFrame(() => {
      for (let i = 0; i < cnt; i++) {
        const c = document.createElement('canvas');
        drawSprite(c, cls, pal, String(Date.now() + i));
        grid.appendChild(c);
      }
      status.textContent = `${cnt} ${cls} — procedural demo`;
      goBtn.disabled = false;
    });
  });

  goBtn.click();
}

// ── Beta section — real WebGPU pipeline via pkg/ ──────────────────────────
//
// On click: load wasm-bindgen module, init device, fetch model bytes, hand
// them to the rust runtime. Then bind the in-frame Generate button to call
// generate(class, seed, steps) and display the returned PNG.

const BETA_CLASSES = [
  // 16 hand-picked classes that the model has the most coverage on.
  // Matches the desktop CLI defaults; not exhaustive.
  'character', 'knight', 'goblin', 'skeleton',
  'dragon', 'demon', 'slime', 'ghost',
  'sword', 'staff_wand', 'shield', 'potion',
  'tree', 'building', 'food', 'gem_treasure',
];

function initBetaPreflight() {
  // Cheap check on page load. Run before the user clicks anything heavy.
  const warn = document.getElementById('webgpu-warn');
  const launch = document.getElementById('beta-launch');
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    if (warn) warn.hidden = false;
    if (launch) {
      launch.disabled = true;
      launch.textContent = 'WebGPU unavailable';
    }
  }
}

async function initBeta() {
  const btn        = document.getElementById('beta-launch');
  const frame      = document.getElementById('beta-frame');
  const loading    = document.getElementById('beta-loading');
  const loadText   = document.getElementById('beta-loading-text');
  const controls   = document.getElementById('beta-controls');
  const output     = document.getElementById('beta-output');
  const classSel   = document.getElementById('beta-class');
  const stepsIn    = document.getElementById('beta-steps');
  const seedIn     = document.getElementById('beta-seed');
  const goBtn      = document.getElementById('beta-go');
  const img        = document.getElementById('beta-img');
  const stats      = document.getElementById('beta-stats');

  // Populate class dropdown.
  for (const c of BETA_CLASSES) {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    classSel.appendChild(opt);
  }

  let runtime = null; // imported pkg, set after first load

  btn.addEventListener('click', async () => {
    if (!navigator.gpu) {
      document.getElementById('webgpu-warn').hidden = false;
      return;
    }
    btn.hidden = true;
    frame.hidden = false;

    try {
      loadText.textContent = 'Loading runtime…';
      const mod = await import('./pkg/pixel_forge_wasm.js');
      await mod.default();
      mod.init_panic_hook();

      if (!mod.webgpu_available()) {
        throw new Error('navigator.gpu disappeared after wasm init');
      }

      loadText.textContent = 'Booting GPU…';
      const banner = await mod.boot();

      loadText.textContent = 'Fetching model (2.2 MB)…';
      const resp = await fetch('./models/cinder.safetensors');
      if (!resp.ok) throw new Error(`model fetch ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());

      loadText.textContent = 'Uploading weights…';
      const param_count = mod.load_model(bytes);

      // Optional sidecar: when the model was trained with z-score
      // normalization, the manifest sits next to the safetensors.
      // 404 = no normalizer; just stay on the legacy [0,1] path.
      const nrmResp = await fetch('./models/cinder.safetensors.normalize.json');
      if (nrmResp.ok) {
        mod.set_normalizer(new Uint8Array(await nrmResp.arrayBuffer()));
      }

      runtime = mod;
      loading.hidden = true;
      controls.hidden = false;
      stats.textContent = `Ready — ${banner} · ${(param_count / 1e6).toFixed(2)}M weights`;
      output.hidden = false;
    } catch (err) {
      loading.innerHTML =
        '<p style="color:var(--orange)">Beta failed to load:<br><code>' +
        String(err).replace(/</g, '&lt;') +
        '</code></p>';
    }
  });

  goBtn.addEventListener('click', async () => {
    if (!runtime) return;
    goBtn.disabled = true;
    const cls = classSel.value;
    const seed = parseInt(seedIn.value, 10) >>> 0;
    const steps = Math.max(4, Math.min(80, parseInt(stepsIn.value, 10) || 40));
    stats.textContent = `Generating ${cls}…`;
    const t0 = performance.now();
    try {
      const png = await runtime.generate(cls, seed, steps);
      const blob = new Blob([png], { type: 'image/png' });
      img.src = URL.createObjectURL(blob);
      const ms = (performance.now() - t0).toFixed(0);
      stats.textContent = `${cls} · seed ${seed} · ${steps} steps · ${ms} ms`;
    } catch (err) {
      stats.textContent = 'Generate failed: ' + String(err);
    } finally {
      goBtn.disabled = false;
    }
  });
}

// ── Upload section ────────────────────────
function initUpload() {
  const dz      = document.getElementById('dropzone');
  const fileIn  = document.getElementById('file');
  const prev    = document.getElementById('preview');
  const inner   = dz.querySelector('.dropzone-inner');
  const form    = document.getElementById('upload-form');
  const submit  = document.getElementById('submit-btn');
  const msg     = document.getElementById('upload-msg');
  let selected  = null;

  function setMsg(text, err = false) {
    msg.textContent = text;
    msg.className = 'upload-msg' + (err ? ' error' : '');
  }

  function validate() {
    submit.disabled = !(
      selected &&
      document.getElementById('meta-orig').checked &&
      document.getElementById('meta-clean').checked
    );
  }

  function loadFile(file) {
    if (!file || file.type !== 'image/png') { setMsg('PNG files only.', true); return; }
    if (file.size > 256 * 1024) { setMsg('File must be under 256 KB.', true); return; }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (img.width > 256 || img.height > 256) {
        setMsg('Image must be ≤256×256.', true);
        URL.revokeObjectURL(url);
        return;
      }
      prev.src = url;
      prev.hidden = false;
      inner.hidden = true;
      selected = file;
      setMsg('');
      validate();
    };
    img.onerror = () => { setMsg('Could not read image.', true); URL.revokeObjectURL(url); };
    img.src = url;
  }

  dz.addEventListener('click', () => fileIn.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileIn.click(); });
  fileIn.addEventListener('change', () => { if (fileIn.files[0]) loadFile(fileIn.files[0]); });

  dz.addEventListener('dragover',  (e) => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', ()  => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    loadFile(e.dataTransfer.files[0]);
  });

  document.getElementById('meta-orig').addEventListener('change', validate);
  document.getElementById('meta-clean').addEventListener('change', validate);

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!selected) return;
    const reader = new FileReader();
    reader.onload = () => {
      const entry = {
        id:      Date.now(),
        title:   (document.getElementById('meta-title').value  || selected.name).slice(0, 64),
        author:  (document.getElementById('meta-author').value || 'anonymous').slice(0, 64),
        cls:     document.getElementById('meta-class').value,
        license: document.getElementById('meta-license').value,
        dataUrl: reader.result,
        ts:      new Date().toISOString(),
      };
      const stored = JSON.parse(localStorage.getItem('pf-gallery') || '[]');
      stored.unshift(entry);
      localStorage.setItem('pf-gallery', JSON.stringify(stored.slice(0, 200)));
      setMsg('Sprite added to gallery.');
      refreshGallery();
      form.reset();
      prev.hidden = true;
      inner.hidden = false;
      selected = null;
      submit.disabled = true;
    };
    reader.readAsDataURL(selected);
  });
}

// ── Gallery section ───────────────────────
function refreshGallery() {
  const grid  = document.getElementById('gallery-grid');
  const empty = document.getElementById('gallery-empty');
  const items = JSON.parse(localStorage.getItem('pf-gallery') || '[]');

  if (items.length === 0) {
    grid.hidden = true;
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  grid.hidden = false;
  grid.innerHTML = items.map(it => `
    <div class="gallery-item">
      <img src="${esc(it.dataUrl)}" alt="${esc(it.title)}" title="${esc(it.author)} — ${esc(it.cls)}">
      <span>${esc(it.title)}</span>
    </div>
  `).join('');
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Boot ──────────────────────────────────
initGenerate();
initBetaPreflight();
initBeta();
initUpload();
refreshGallery();
