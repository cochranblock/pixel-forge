// Unlicense — cochranblock.org

// ── Palettes ──────────────────────────────
const PALETTES = {
  stardew:   ['#1a1a2e','#2d5a1b','#5a8a34','#8ab464','#c8e8a0','#2e4a6e','#4a7aaa','#7aaad4','#e8d080','#d08040','#a04020','#e8a060','#f0e0c0','#ffffff','#808080','#404050'],
  starbound:  ['#0d0d1a','#1a2a4a','#2a4a8a','#4a8aaa','#8aaac8','#c8d8e8','#e8eef8','#ffffff','#c85020','#e87030','#f0a040','#f0d060','#20a040','#40c060','#60e080','#80f0a0'],
  snes:      ['#000000','#1c1c1c','#484848','#787878','#a8a8a8','#d8d8d8','#ffffff','#580000','#900000','#c80000','#ff0000','#ff6060','#004400','#006800','#009000','#00c000'],
  nes:       ['#000000','#fcfcfc','#f8f8f8','#bcbcbc','#7c7c7c','#a4e4fc','#3cbcfc','#0078f8','#0000fc','#b8b8f8','#6888fc','#0058f8','#0000bc','#d8b8f8','#9878f8','#6844fc'],
  gameboy:   ['#0f380f','#306230','#8bac0f','#9bbc0f'],
  pico8:     ['#000000','#1d2b53','#7e2553','#008751','#ab5236','#5f574f','#c2c3c7','#fff1e8','#ff004d','#ffa300','#ffec27','#00e436','#29adff','#83769c','#ff77a8','#ffccaa'],
  endesga32: ['#be4a2f','#d77643','#ead4aa','#e4a672','#b86f50','#733e39','#3e2731','#a22633','#e43b44','#f77622','#feae34','#fee761','#63c74d','#3e8948','#265c42','#193c3e','#124e89','#0099db','#2ce8f5','#ffffff','#c0cbdc','#8b9bb4','#5a6988','#3a4466','#262b44','#181425','#ff0044','#68386c','#b55088','#f6757a','#e8b796','#c28569'],
};

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

  // Sample 3 colors spread across the palette
  const base = Math.floor(rng() * pal.length);
  const n = pal.length;
  const colors = [
    null,
    pal[base % n],
    pal[(base + Math.floor(n * 0.3)) % n],
    pal[(base + Math.floor(n * 0.6)) % n],
  ];

  // Fill left half using class-specific silhouette
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < half; x++) {
      const nx = x / half;
      const ny = y / S;
      const d = shapeDensity(cls, nx, ny);
      if (d <= 0 || rng() > d) continue;
      const c = rng();
      px[y * S + x] = c < 0.50 ? 1 : c < 0.85 ? 2 : 3;
    }
  }

  // Mirror left→right
  for (let y = 0; y < S; y++)
    for (let x = 0; x < half; x++)
      px[y * S + (S - 1 - x)] = px[y * S + x];

  // Render
  ctx.clearRect(0, 0, S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const c = px[y * S + x];
      if (!c) continue;
      ctx.fillStyle = colors[c];
      ctx.fillRect(x, y, 1, 1);
    }
  }
}

// ── Generate section ──────────────────────
function initGenerate() {
  const clsEl  = document.getElementById('gen-class');
  const palEl  = document.getElementById('gen-palette');
  const cntEl  = document.getElementById('gen-count');
  const cntOut = document.getElementById('gen-count-out');
  const goBtn  = document.getElementById('gen-go');
  const status = document.getElementById('gen-status');
  const grid   = document.getElementById('gen-grid');

  cntEl.addEventListener('input', () => { cntOut.value = cntEl.value; });

  goBtn.addEventListener('click', () => {
    const cls = clsEl.value;
    const pal = palEl.value;
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

// ── Beta section ──────────────────────────
function initBeta() {
  const btn   = document.getElementById('beta-launch');
  const frame = document.getElementById('beta-frame');

  btn.addEventListener('click', () => {
    frame.hidden = false;
    btn.hidden = true;
    // Attempt to load the WASM binary if deployed alongside this page.
    // Falls back gracefully — the spinner stays visible until init() resolves.
    const s = document.createElement('script');
    s.type = 'module';
    s.textContent = `
      import init from './pixel_forge.js';
      init('./pixel_forge_bg.wasm')
        .then(() => {
          document.getElementById('beta-loading').hidden = true;
        })
        .catch(() => {
          const el = document.getElementById('beta-loading');
          el.innerHTML = '<p style="color:var(--orange)">Beta build not deployed yet.<br>Run <code>cargo build --release -p pixel-forge</code> and serve from <code>web/</code>.</p>';
        });
    `;
    document.head.appendChild(s);
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
initBeta();
initUpload();
refreshGallery();
