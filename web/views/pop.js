// Base for population models (records 16–21): firing meters, a metric chart, a
// probe trace, the decision, weight fields and a raster with per-step scalars.
// A model view calls popView({...}) with its own Fig. 01 and registers the result.
"use strict";
const ROLE = { input: 0, latent: 1, prediction: 2, error: 3, label: 4, output: 5 };
const ORANGE = "#ec835a";
const SCALAR_COLORS = [G[4], ORANGE, G.p, "#7fb2e5"];

// signed byte (128 = 0) → orange below zero, board at zero, phthalo to pale above
const DIVERGE = (() => {
  const neg = [[0, 18, 53, 36], [0.5, 120, 74, 50], [1, 236, 131, 90]];
  const pos = [[0, 18, 53, 36], [0.45, 17, 134, 92], [0.75, 82, 196, 149], [1, 238, 242, 238]];
  const ramp = (stops, t) => {
    t = Math.min(1, t);
    let k = 1; while (stops[k][0] < t) k++;
    const [t0, ...c0] = stops[k - 1], [t1, ...c1] = stops[k], f = (t - t0) / (t1 - t0);
    return c0.map((c, i) => c + (c1[i] - c) * f);
  };
  const out = new Uint8ClampedArray(256 * 3);
  for (let b = 0; b < 256; b++) {
    const c = b >= 128 ? ramp(pos, (b - 128) / 127) : ramp(neg, (128 - b) / 127);
    out.set(c, b * 3);
  }
  return out;
})();

// draws a w×h grid of cells; fill(i) returns [r, g, b] or null to skip
function drawGrid(x, ox, oy, size, gw, gh, fill) {
  const cs = size / Math.max(gw, gh);
  const im = x.createImageData(gw, gh);
  for (let i = 0; i < gw * gh; i++) {
    const c = fill(i);
    if (c) { im.data[i * 4] = c[0]; im.data[i * 4 + 1] = c[1]; im.data[i * 4 + 2] = c[2]; im.data[i * 4 + 3] = 255; }
  }
  const tmp = document.createElement("canvas");
  tmp.width = gw; tmp.height = gh;
  tmp.getContext("2d").putImageData(im, 0, 0);
  x.imageSmoothingEnabled = false;
  x.drawImage(tmp, ox, oy, gw * cs, gh * cs);
  return cs;
}

function popView(model) {
  const P = {
    series: { step: [], probe: [], image: [] }, labelPop: -1,
    glow: [], count: [], t: 0, ring: [], ringAt: 0,
    ptrace: null, pspk: null, strace: null, probe: { pop: 1, unit: 0, th: 1 },
    outcome: null, live: null, fields: null, tiles: null, metric: [], geom: {},
  };
  S.pop = P;
  const pops = () => S.pops;
  const unitName = (pop, unit) => `${(pops()[pop] || { name: "?" }).name} · N${String(unit).padStart(2, "0")}`;

  const markup = {
    firing: `
      <div class="k">Firing · Hz</div>
      <div id="meters"></div>
      <div class="k" id="metric-k" style="margin-top:14px"></div>
      <div style="height:96px;margin-top:6px"><canvas id="metric"></canvas></div>
    `,
    center: `
      <div class="bp"><span class="k">${model.fig1}</span><canvas id="fig1" style="cursor:crosshair"></canvas></div>
      <div class="bp"><span class="k" id="fig2-k">Fig. 02 — fields</span><canvas id="fields" style="cursor:pointer"></canvas></div>
    `,
    right: `
      <div>
        <div class="k">Probe</div>
        <div class="num probe-name" id="probe-v" style="font-size:34px">–</div>
      </div>
      <div class="bp" style="height:110px"><canvas id="trace"></canvas></div>
      <div class="pair" id="pvars"></div>
      <div class="decision">
        <span class="k">Decision</span>
        <div class="verdict">
          <span class="num digit" id="digit-v">–</span>
          <div class="truth"><span id="truth-v"></span><span class="badge" id="badge-v"></span></div>
        </div>
        <div style="flex:1;min-height:90px"><canvas id="classes"></canvas></div>
      </div>
    `,
    bottom: `<span class="k">Fig. 03 — raster · step scalars</span><canvas id="raster"></canvas>`,
  };

  // ---- records -------------------------------------------------------------------------------
  const H = {
    16(v, p) { // SERIES
      let o = p;
      for (const key of ["step", "probe", "image"]) {
        const n = v.getUint8(o++), names = [];
        for (let k = 0; k < n; k++) { let s; [s, o] = str(v, o); names.push(s); }
        P.series[key] = names;
      }
      $("xstats").replaceChildren(...P.series.image.map((name, i) => {
        const d = document.createElement("div"), k = document.createElement("span"), n = document.createElement("span");
        d.className = "stat"; k.className = "k"; k.textContent = name; n.className = "num"; n.id = `x${i}-v`; n.textContent = "–";
        d.append(k, n); return d;
      }));
      $("metric-k").textContent = P.series.image[0] || "";
      $("pvars").replaceChildren(...P.series.probe.map((name, i) => {
        const d = document.createElement("div"), k = document.createElement("span"), n = document.createElement("span");
        k.className = "k"; k.textContent = name; n.className = "num"; n.id = `pv${i}`; n.textContent = "–";
        d.append(k, n); return d;
      }));
      alloc();
    },
    17(v, p) { // SPIKES
      let t = v.getUint16(p, true), o = p + 4;
      const n = v.getUint16(p + 2, true), nS = P.series.step.length, nP = P.series.probe.length, ps = pops();
      for (let k = 0; k < n; k++, t++) {
        const lists = [];
        for (let q = 0; q < ps.length; q++) {
          const na = v.getUint16(o, true); o += 2;
          const a = new Uint16Array(na), g = P.glow[q];
          for (let i = 0; i < na; i++, o += 2) { a[i] = v.getUint16(o, true); g[a[i]] = 1; }
          P.count[q] += na;
          if (q === P.probe.pop && a.includes(P.probe.unit) && t <= S.T) P.pspk[t] = 1;
          if (q === P.labelPop) for (const i of a) P.live[i]++;
          lists.push(a);
        }
        const sc = new Float32Array(nS);
        for (let i = 0; i < nS; i++, o += 4) sc[i] = v.getFloat32(o, true);
        for (let i = 0; i < nP; i++, o += 4) if (t <= S.T) P.ptrace[t * nP + i] = v.getFloat32(o, true);
        if (t <= S.T) P.strace.set(sc, t * nS);
        P.t = t;
        if (model.step) model.step(t, lists, sc);
        P.ring[P.ringAt] = { lists, sc };
        P.ringAt = (P.ringAt + 1) % P.ring.length;
      }
      dirty("net", "trace", "raster", "classes", "decision", "meters");
    },
    18(v, p) { // PROBE
      P.probe = { pop: v.getUint8(p), unit: v.getUint16(p + 1, true), th: v.getFloat32(p + 3, true) };
      dirty("trace", "fields", "net");
    },
    19(v, p) { // OUTCOME
      const o = { predicted: v.getUint8(p + 4), correct: v.getUint8(p + 5) === 1, learn: v.getUint8(p + 6) === 1, acc: v.getFloat32(p + 7, true) };
      let q = p + 11;
      const list = () => { const n = v.getUint8(q++), a = new Float32Array(n); for (let i = 0; i < n; i++, q += 4) a[i] = v.getFloat32(q, true); return a; };
      o.scores = list(); o.scalars = list(); o.smoothed = list();
      const ns = v.getUint8(q++); o.spikes = [];
      for (let i = 0; i < ns; i++, q += 4) o.spikes.push(v.getUint32(q, true));
      o.label = S.label;
      P.outcome = o;
      o.smoothed.forEach((m, i) => { const el = $(`x${i}-v`); if (el) el.textContent = fmtNum(m); });
      const n = S.progress ? S.progress.images : 0, last = P.metric[P.metric.length - 1];
      if (o.learn && o.smoothed.length && (!last || last[0] < n)) {
        P.metric.push([n, o.smoothed[0]]);
        if (P.metric.length > 4000) P.metric = P.metric.filter((_, i) => i % 2 === 0);
      }
      if (model.outcome) model.outcome(o);
      dirty("decision", "classes", "metric", "net");
    },
    20(v, p) { // FIELDS
      let o;
      const f = {};
      [f.title, o] = str(v, p);
      f.pop = v.getUint8(o); f.n = v.getUint16(o + 1, true); f.w = v.getUint16(o + 3, true); f.h = v.getUint16(o + 5, true);
      f.ch = v.getUint8(o + 7); f.scale = v.getFloat32(o + 8, true); o += 12;
      f.units = new Uint16Array(f.n); f.labels = new Uint8Array(f.n); f.values = new Float32Array(f.n);
      for (let k = 0; k < f.n; k++, o += 7) { f.units[k] = v.getUint16(o, true); f.labels[k] = v.getUint8(o + 2); f.values[k] = v.getFloat32(o + 3, true); }
      f.data = new Uint8Array(v.buffer.slice(v.byteOffset + o, v.byteOffset + o + f.n * f.w * f.h * f.ch));
      P.fields = f; P.tiles = null;
      $("fig2-k").textContent = f.title;
      if (model.fields) model.fields(f);
      dirty("fields");
    },
    21(v, p) { // MAP
      const m = { w: v.getUint16(p + 1, true), h: v.getUint16(p + 3, true), signed: v.getUint8(p + 5) === 1, scale: v.getFloat32(p + 6, true) };
      m.data = new Uint8Array(v.buffer.slice(v.byteOffset + p + 10, v.byteOffset + p + 10 + m.w * m.h));
      if (model.map) model.map(v.getUint8(p), m);
      dirty("net");
    },
  };

  function alloc() {
    const T = S.T, nP = Math.max(1, P.series.probe.length), nS = P.series.step.length;
    P.ptrace = new Float32Array((T + 1) * nP); P.pspk = new Uint8Array(T + 1); P.strace = new Float32Array((T + 1) * Math.max(1, nS));
  }

  // ---- panels ------------------------------------------------------------------------------
  const fmtNum = (x) => (Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : +x.toPrecision(3));

  function renderMeters() {
    const ps = pops(), dt = S.meta.dt || 1, t = Math.max(1, P.t);
    if ($("meters").childElementCount !== ps.length) {
      $("meters").replaceChildren(...ps.map((q, i) => {
        const d = document.createElement("div");
        d.className = "meter";
        d.innerHTML = `<div><span></span><span class="num" id="hz${i}">0</span></div><div class="track"><div class="fill" id="bar${i}"></div></div>`;
        d.firstChild.firstChild.textContent = q.name;
        return d;
      }));
    }
    ps.forEach((q, i) => {
      const hz = P.count[i] / q.size / (t * dt) * 1000;
      $(`hz${i}`).textContent = hz.toFixed(1);
      $(`bar${i}`).style.width = Math.min(100, hz / 64 * 100) + "%";
    });
  }

  function renderProbe() {
    const { pop, unit } = P.probe, nP = P.series.probe.length, t = Math.max(1, Math.min(P.t, S.T));
    $("probe-v").textContent = unitName(pop, unit);
    for (let i = 0; i < nP; i++) { const el = $(`pv${i}`); if (el) el.textContent = fmtNum(P.ptrace[t * nP + i]); }
  }

  function drawTrace() {
    const g = fit($("trace"));
    if (!g || !P.ptrace) return;
    const { x, w, h } = g, T = S.T, nP = Math.max(1, P.series.probe.length), th = P.probe.th, end = Math.min(P.t, T);
    const X = (t) => ((t - 1) / Math.max(1, T - 1)) * w;
    x.fillStyle = pa(0.06); for (let k = 1; k < 4; k++) x.fillRect(0, Math.round((h * k) / 4), w, 1);
    // secondary variables first, each on its own scale
    for (let i = nP - 1; i >= 0; i--) {
      let lo = i === 0 ? Math.min(0, -0.25 * th) : Infinity, hi = i === 0 ? 1.15 * th : -Infinity;
      for (let t = 1; t <= end; t++) { const v = P.ptrace[t * nP + i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      if (!(hi > lo)) { lo -= 1; hi += 1; }
      const Y = (v) => h - 6 - ((v - lo) / (hi - lo)) * (h - 18);
      if (i === 0) {
        x.setLineDash([4, 4]); x.strokeStyle = pa(0.5); x.lineWidth = 1;
        x.beginPath(); x.moveTo(0, Y(th)); x.lineTo(w, Y(th)); x.stroke(); x.setLineDash([]);
        x.fillStyle = G.p; x.font = `600 10px ${FH}`; x.textAlign = "right"; x.fillText("V_TH", w - 6, Y(th) - 4);
      }
      x.strokeStyle = i === 0 ? G[3] : i === 1 ? gb(0.55) : "rgba(236,131,90,0.55)"; x.lineWidth = i === 0 ? 1.5 : 1;
      x.beginPath();
      for (let t = 1; t <= end; t++) { const px = X(t), py = Y(P.ptrace[t * nP + i]); if (t === 1) x.moveTo(px, py); else x.lineTo(px, py); }
      x.stroke();
    }
    x.strokeStyle = G.p; x.lineWidth = 1; x.beginPath();
    for (let t = 1; t <= end; t++) if (P.pspk[t]) { x.moveTo(X(t), 3); x.lineTo(X(t), 14); }
    x.stroke();
  }

  function renderDecision() {
    const o = P.outcome, live = argmax(P.live);
    const pred = o ? o.predicted : live;
    $("digit-v").textContent = pred === NONE || pred < 0 ? "–" : pred;
    const label = o ? o.label : S.label;
    $("truth-v").textContent = label >= 0 ? `label ${label}` + (S.learn ? "" : " · test") : "";
    const b = $("badge-v");
    if (o) { b.className = "badge " + (o.correct ? "ok" : "miss"); b.textContent = o.correct ? "✓ correct" : o.predicted === NONE ? "✗ no answer" : "✗ wrong"; }
    else { b.className = "badge"; b.textContent = ""; }
  }
  const argmax = (a) => { if (!a) return -1; let b = -1, m = 0; a.forEach((v, i) => { if (v > m) { m = v; b = i; } }); return b; };

  function drawClasses() {
    const g = fit($("classes"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, C = S.meta.classes, o = P.outcome;
    const ev = o && o.scores.length === C ? o.scores : P.live || new Float32Array(C);
    let lo = Math.min(0, ...ev), hi = Math.max(...ev);
    if (!(hi > lo)) hi = lo + 1;
    const band = w / C, bw = Math.min(24, band - 6), base = h - 18, pred = o ? o.predicted : argmax(P.live);
    P.geom.classes = { band, ev };
    x.fillStyle = pa(0.18); x.fillRect(0, base, w, 1);
    for (let c = 0; c < C; c++) {
      const cx = band * c + band / 2, bh = ((ev[c] - lo) / (hi - lo)) * (base - 4);
      if (bh > 0.5) { x.fillStyle = c === pred ? G[2] : G[6]; x.fillRect(cx - bw / 2, base - bh, bw, bh); }
      const label = o ? o.label : S.label;
      x.font = `${c === label ? 600 : 400} 13px ${FH}`; x.textAlign = "center"; x.fillStyle = c === label ? G.p : G[2];
      x.fillText(c, cx, h - 3);
    }
  }

  function buildTiles(f) {
    const c = document.createElement("canvas"), n = f.w * f.h;
    c.width = f.w * f.n; c.height = f.h;
    const x = c.getContext("2d"), im = x.createImageData(f.w * f.n, f.h);
    for (let k = 0; k < f.n; k++) for (let i = 0; i < n; i++) {
      const px = ((i / f.w) | 0) * f.w * f.n + k * f.w + (i % f.w), base = k * n * f.ch;
      let r, g2, b;
      if (f.ch === 2) { // ON green, OFF red, positive parts only
        const on = Math.max(0, f.data[base + i] - 128) * 2, off = Math.max(0, f.data[base + n + i] - 128) * 2;
        r = 18 + off * 0.86; g2 = 53 + on * 0.76; b = 36;
      } else {
        const q = f.data[base + i] * 3; r = DIVERGE[q]; g2 = DIVERGE[q + 1]; b = DIVERGE[q + 2];
      }
      im.data[px * 4] = r; im.data[px * 4 + 1] = g2; im.data[px * 4 + 2] = b; im.data[px * 4 + 3] = 255;
    }
    x.putImageData(im, 0, 0);
    return c;
  }

  function drawFields() {
    const g = fit($("fields")), f = P.fields;
    if (!g || !f) return;
    const { x, w, h } = g, N = f.n, top = 30, pad = 12;
    if (!P.tiles) P.tiles = buildTiles(f);
    let best = { s: 0 };
    for (let cols = 1; cols <= N; cols++) {
      const rows = Math.ceil(N / cols), s = Math.min((w - 2 * pad) / cols, (h - top - pad) / rows);
      if (s > best.s) best = { s, cols };
    }
    const { s, cols } = best, t = s - Math.max(2, s * 0.08), ox = (w - cols * s) / 2;
    P.geom.fields = { ox, top, s, t, cols };
    x.imageSmoothingEnabled = false;
    for (let k = 0; k < N; k++) {
      const tx = ox + (k % cols) * s, ty = top + Math.floor(k / cols) * s;
      x.drawImage(P.tiles, k * f.w, 0, f.w, f.h, tx, ty, t, t * f.h / f.w);
      if (f.labels[k] !== NONE && t >= 22) {
        x.font = `600 ${Math.min(13, t * 0.22)}px ${FH}`; x.textAlign = "left"; x.fillStyle = pa(0.75);
        x.fillText(f.labels[k], tx + 3, ty + Math.min(13, t * 0.22) + 1);
      }
      const val = Math.max(-1, Math.min(1, f.values[k]));
      if (t >= 14) { x.fillStyle = val >= 0 ? gb(0.9) : ORANGE; x.fillRect(tx, ty + t - 2, Math.abs(val) * t, 2); }
      if (f.pop === P.probe.pop && f.units[k] === P.probe.unit) { x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, tx + t / 2, ty + t / 2, t / 2 + 3, 5); }
    }
  }

  function drawRaster() {
    const g = fit($("raster"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, ps = pops(), L = P.ring.length, L0 = 64, top = 18, H0 = h - top - 4;
    const weights = ps.map((q) => Math.max(1.5, Math.sqrt(q.size))), sum = weights.reduce((a, b) => a + b, 0);
    const rows = []; let y = top;
    ps.forEach((q, i) => { const hh = (weights[i] / sum) * H0; rows.push([y, hh]); y += hh; });
    const cw = (w - L0 - 8) / L;
    x.font = `600 11px ${FH}`; x.textAlign = "left";
    ps.forEach((q, i) => {
      const [ry, rh] = rows[i];
      x.fillStyle = q.role === ROLE.input ? G[3] : G.p; x.fillText(q.name.toUpperCase(), 8, ry + rh / 2 + 4);
      if (i) { x.fillStyle = pa(0.1); x.fillRect(L0, Math.round(ry), w - L0, 1); }
    });
    const nS = P.series.step.length, maxS = new Float32Array(nS);
    for (const e of P.ring) if (e && e.sc) for (let i = 0; i < nS; i++) maxS[i] = Math.max(maxS[i], Math.abs(e.sc[i]));
    for (let k = 0; k < L; k++) {
      const e = P.ring[(P.ringAt + k) % L];
      if (!e) continue;
      const px = L0 + k * cw;
      if (e.mark !== undefined) { x.fillStyle = pa(0.25); x.fillRect(Math.round(px) + 0.5, top, 1, h - top); x.fillStyle = G[2]; x.fillText(e.mark, px + 4, top + 11); continue; }
      e.lists.forEach((a, i) => {
        const [ry, rh] = rows[i], ru = rh / ps[i].size;
        x.fillStyle = ps[i].role === ROLE.input ? ga(0.85) : ps[i].role === ROLE.prediction ? gb(0.9) : G.p;
        for (const u of a) x.fillRect(px, ry + u * ru, Math.max(1.2, cw), Math.max(1, ru));
      });
    }
    // step scalars as lines over the raster, each scaled to its own peak
    for (let i = 0; i < nS; i++) {
      if (!(maxS[i] > 0)) continue;
      x.strokeStyle = SCALAR_COLORS[i % SCALAR_COLORS.length]; x.lineWidth = 1.5; x.beginPath();
      let first = true;
      for (let k = 0; k < L; k++) {
        const e = P.ring[(P.ringAt + k) % L];
        if (!e || !e.sc) { first = true; continue; }
        const px = L0 + k * cw, py = top + H0 - (Math.abs(e.sc[i]) / maxS[i]) * (H0 - 4);
        if (first) x.moveTo(px, py); else x.lineTo(px, py);
        first = false;
      }
      x.stroke();
      x.fillStyle = SCALAR_COLORS[i % SCALAR_COLORS.length]; x.textAlign = "right";
      x.fillText(P.series.step[i], w - 14 - 70 * (nS - 1 - i), 12);
    }
    x.fillStyle = G.p; x.fillRect(w - 8, top, 1, h - top);
  }

  function drawMetric() {
    const g = fit($("metric"));
    if (!g) return;
    const { x, w, h } = g, pts = P.metric, L0 = 34, b = h - 4, top = 6;
    if (pts.length < 2) return;
    let lo = Infinity, hi = -Infinity;
    for (const [, v] of pts) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!(hi > lo)) { hi = lo + 1; }
    const maxX = pts[pts.length - 1][0] || 1, X = (n) => L0 + (n / maxX) * (w - L0 - 6), Y = (v) => b - ((v - lo) / (hi - lo)) * (b - top);
    x.font = `11px ${FH}`; x.textAlign = "right"; x.fillStyle = G[2];
    for (const v of [lo, hi]) { x.fillStyle = pa(0.12); x.fillRect(L0, Math.round(Y(v)), w - L0, 1); x.fillStyle = G[2]; x.fillText(fmtNum(v), L0 - 5, Y(v) + 4); }
    x.beginPath(); x.moveTo(X(pts[0][0]), Y(pts[0][1]));
    for (const [n, v] of pts) x.lineTo(X(n), Y(v));
    x.lineWidth = 1.5; x.strokeStyle = ORANGE; x.stroke();
  }

  // ---- interaction ---------------------------------------------------------------------------
  function bind() {
    $("fig1").onclick = (e) => {
      const hit = model.hit && model.hit(...local(e));
      if (hit) send(`probe ${hit[0]} ${hit[1]}`);
    };
    $("fig1").onpointermove = (e) => {
      const tip = model.tip && model.tip(...local(e));
      if (tip) showTip(e, tip[0], tip[1]); else hideTip();
    };
    $("fig1").onpointerleave = hideTip;
    const fieldHit = (e) => {
      const gm = P.geom.fields, f = P.fields;
      if (!gm || !f) return -1;
      const [mx, my] = local(e), c = Math.floor((mx - gm.ox) / gm.s), r = Math.floor((my - gm.top) / gm.s), k = r * gm.cols + c;
      return c >= 0 && c < gm.cols && r >= 0 && k < f.n ? k : -1;
    };
    $("fields").onclick = (e) => { const k = fieldHit(e); if (k >= 0) send(`probe ${P.fields.pop} ${P.fields.units[k]}`); };
    $("fields").onpointermove = (e) => {
      const k = fieldHit(e), f = P.fields;
      if (k < 0) return hideTip();
      showTip(e, unitName(f.pop, f.units[k]), ` · label ${f.labels[k] === NONE ? "–" : f.labels[k]} · ${fmtNum(f.values[k])}`);
    };
    $("fields").onpointerleave = hideTip;
    addEventListener("keydown", (e) => { if (e.key === "Escape") send("probe auto"); });
  }

  return {
    layout() {
      for (const [id, html] of Object.entries(markup)) $(id).innerHTML = html;
      bind();
      if (model.layout) model.layout(P);
    },
    pops() {
      const ps = pops();
      P.glow = ps.map((q) => new Float32Array(q.size)); P.count = ps.map(() => 0);
      P.labelPop = ps.findIndex((q) => (q.role === ROLE.label || q.role === ROLE.output) && q.size === S.meta.classes);
      P.live = P.labelPop >= 0 ? new Float32Array(S.meta.classes) : null;
      P.ring = new Array(S.T * 3).fill(null); P.ringAt = 0;
      alloc();
      if (model.pops) model.pops(P);
    },
    image() {
      // without live label spikes the last decision stays up until the next one
      P.count.fill(0); P.t = 0;
      if (P.live) P.outcome = null;
      if (P.live) P.live.fill(0);
      if (P.ptrace) { P.ptrace.fill(0); P.pspk.fill(0); P.strace.fill(0); }
      P.ring[P.ringAt] = { mark: S.label };
      P.ringAt = (P.ringAt + 1) % P.ring.length;
      if (model.image) model.image(P);
      dirty("net", "trace", "decision", "classes", "meters");
    },
    history(series, pts) {
      if (series === 1) { P.metric = pts; dirty("metric"); }
    },
    H: Object.assign({}, H, model.H),
    stats() {},
    frame(all) {
      let glowing = false;
      for (const g of P.glow) for (let i = 0; i < g.length; i++) if (g[i] > 0.01) { g[i] *= 0.88; glowing = true; }
      if (glowing) dirty("net");
      if (all || S.dirty.has("net")) { const g = fit($("fig1")); if (g) { P.geom.fig1 = g; model.draw(g.x, g.w, g.h, P); } }
      if (all || S.dirty.has("fields")) drawFields();
      if (all || S.dirty.has("trace")) { drawTrace(); renderProbe(); }
      if (all || S.dirty.has("classes")) drawClasses();
      if (all || S.dirty.has("decision")) renderDecision();
      if (all || S.dirty.has("raster")) drawRaster();
      if (all || S.dirty.has("meters")) renderMeters();
      if (all || S.dirty.has("metric")) drawMetric();
    },
  };
}

// Default figure: every population as a grid of cells lit by recent spikes,
// left to right in POPS order. Model views can call it for part of their figure.
function drawPops(x, w, h, P, only) {
  const ps = S.pops.map((q, i) => [q, i]).filter(([q, i]) => !only || only.includes(i));
  // grids start below the panel title, whatever their size
  const pad = 28, top = 56, slot = (w - 2 * pad) / ps.length, hits = [];
  rulers(x, w, h);
  ps.forEach(([q, i], k) => {
    const size = Math.min(slot * 0.8, h - top - pad), ox = pad + k * slot + (slot - size) / 2, oy = top + (h - top - pad - size) / 2;
    const glow = P.glow[i];
    const cs = drawGrid(x, ox, oy, size, q.w, q.h, (u) => {
      if (u >= q.size) return null;
      const gl = glow[u];
      return gl > 0.02 ? [18 + 220 * gl, 53 + 189 * gl, 36 + 202 * gl] : [24, 66, 46];
    });
    x.font = `600 12px ${FH}`; x.fillStyle = G[3]; x.textAlign = "left";
    x.fillText(`${q.name.toUpperCase()} ${q.w}×${q.h}`, ox, oy - 8);
    if (i === P.probe.pop) {
      const u = P.probe.unit, cx = ox + ((u % q.w) + 0.5) * cs, cy = oy + (((u / q.w) | 0) + 0.5) * cs;
      x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, cx, cy, Math.max(4, cs * 0.8), 3);
    }
    hits.push({ pop: i, ox, oy, cs, q });
  });
  return hits;
}

// unit under (mx, my) in the boxes drawPops returned
function hitPops(boxes, mx, my) {
  for (const b of boxes || []) {
    const c = Math.floor((mx - b.ox) / b.cs), r = Math.floor((my - b.oy) / b.cs), u = r * b.q.w + c;
    if (c >= 0 && c < b.q.w && r >= 0 && u < b.q.size) return [b.pop, u];
  }
  return null;
}

{
  let boxes = null;
  registerView("pop", popView({
    fig1: "Fig. 01 — populations · click to probe",
    draw(x, w, h, P) { boxes = drawPops(x, w, h, P); },
    hit: (mx, my) => hitPops(boxes, mx, my),
  }));
}
