// Deep convolutional SNN view (mnist-dcsnn-rs). Fig. 01: one image's spike
// wave through S1 (DoG ON/OFF at three scales), the L1 and L2 feature maps
// and the decision layer's class evidence. A cell lights at its first spike,
// paler the earlier it came. Neurons that learn from the image are bracketed,
// with the square they silence in every map.
"use strict";
{
  const PHASES = ["STDP on L1", "STDP on L2", "R-STDP on the decision layer"];
  const NAMES = ["S1", "L1", "L2", "L3"];
  const M = { arch: null, rec: null, first: [], boxes: [], bars: null, cvs: [] };
  const NONE_T = 255;

  const H = {
    49(v, p) { // ARCH
      const n = v.getUint8(p), layers = [];
      let o = p + 1;
      for (let i = 0; i < n; i++, o += 14) {
        layers.push({ maps: v.getUint16(o, true), h: v.getUint16(o + 2, true), w: v.getUint16(o + 4, true), kernel: v.getUint8(o + 6),
          th: v.getFloat32(o + 7, true), pool: v.getUint8(o + 11), winners: v.getUint8(o + 12), radius: v.getUint8(o + 13) });
      }
      M.arch = { layers, per: v.getUint8(o) };
      dirty("net");
    },
    50(v, p) { // DCSNN
      const r = { phase: v.getUint8(p), depth: v.getUint8(p + 1), rates: [], winners: [] };
      let o = p + 2;
      for (let i = 0; i < 8; i++, o += 4) r.rates.push(v.getFloat32(o, true));
      const nw = v.getUint8(o++);
      for (let i = 0; i < nw; i++, o += 8) r.winners.push({ layer: v.getUint8(o), unit: v.getUint16(o + 1, true), t: v.getUint8(o + 3), pot: v.getFloat32(o + 4, true) });
      r.conv = [0, 1, 2].map((i) => v.getFloat32(o + 4 * i, true)); o += 12;
      r.steps = v.getUint8(o); r.classes = v.getUint8(o + 1); o += 2;
      r.evidence = new Float32Array(r.steps * r.classes);
      for (let i = 0; i < r.evidence.length; i++, o += 4) r.evidence[i] = v.getFloat32(o, true);
      M.rec = r;
      dirty("net");
    },
  };

  // a unit's (row, column, map) in a position-major layer
  const split = (L, u) => { const pos = Math.floor(u / L.maps); return [Math.floor(pos / L.w), pos % L.w, u % L.maps]; };
  // where map m sits in its block: S1 is scales across, ON over OFF
  const tileOf = (i, m, cols) => (i === 0 ? [m >> 1, m & 1] : [m % cols, Math.floor(m / cols)]);
  const colsOf = (i, L) => (i === 0 ? Math.ceil(L.maps / 2) : Math.ceil(Math.sqrt(L.maps)));

  // one block of maps at a pixel per neuron, 1px apart; cells by spike time
  function render(i, t) {
    const L = M.arch.layers[i], cols = colsOf(i, L), rows = Math.ceil(L.maps / cols);
    const W = cols * (L.w + 1) - 1, Hh = rows * (L.h + 1) - 1, n = L.maps * L.h * L.w;
    let c = M.cvs[i];
    if (!c || c.width !== W || c.height !== Hh) {
      c = M.cvs[i] = document.createElement("canvas"); c.width = W; c.height = Hh;
      const pix = c.pix = new Int32Array(n);
      for (let u = 0; u < n; u++) {
        const [y, xx, m] = split(L, u), [tc, tr] = tileOf(i, m, cols);
        pix[u] = ((tr * (L.h + 1) + y) * W + tc * (L.w + 1) + xx) * 4;
      }
    }
    const x = c.getContext("2d"), im = x.createImageData(W, Hh), d = im.data, first = M.first[i], T = S.T;
    for (let u = 0; u < n; u++) {
      const px = c.pix[u], ft = first ? first[u] : NONE_T;
      let r = 24, g = 66, b = 46;
      if (ft !== NONE_T && ft <= t) {
        // early spikes pale, late ones phthalo
        const f = (ft - 1) / Math.max(1, T - 1);
        r = 238 - 221 * f; g = 242 - 108 * f; b = 238 - 146 * f;
      }
      d[px] = r; d[px + 1] = g; d[px + 2] = b; d[px + 3] = 255;
    }
    x.putImageData(im, 0, 0);
    return { c, cols, rows, W, Hh };
  }

  function label(x, text, px, py, color) { x.font = `600 12px ${FH}`; x.textAlign = "left"; x.fillStyle = color || G[3]; x.fillText(text, px, py); }

  function arrow(x, x0, x1, y, text) {
    x.strokeStyle = G[4]; x.fillStyle = G[4]; x.lineWidth = 1.4;
    x.beginPath(); x.moveTo(x0, y); x.lineTo(x1, y); x.stroke();
    x.beginPath(); x.moveTo(x1, y); x.lineTo(x1 - 6, y - 4); x.lineTo(x1 - 6, y + 4); x.fill();
    if (text) { x.font = `600 10px ${FH}`; x.textAlign = "center"; x.fillText(text, (x0 + x1) / 2, y - 6); }
  }

  function draw(x, w, h, P) {
    rulers(x, w, h);
    M.boxes = []; M.bars = null;
    if (!M.arch || !M.first.length) return;
    const A = M.arch.layers, r = M.rec, depth = r ? r.depth : 3, t = Math.max(0, Math.min(P.t, S.T));
    const pad = 24, top = 72, bottom = 16, gap = 44, stack = 48, barW = Math.min(170, w * 0.18);
    // neurons across and down each block needs, with 1px between tiles
    const dims = [0, 1, 2].map((i) => { const L = A[i], cols = colsOf(i, L), rows = Math.ceil(L.maps / cols); return [cols * (L.w + 1) - 1, rows * (L.h + 1) - 1]; });
    // S1 sits over L1, then L2, then the decision bars
    const across = Math.max(dims[0][0], dims[1][0]) + dims[2][0], down = Math.max(dims[0][1] + dims[1][1], dims[2][1]);
    const u = Math.max(0.5, Math.min((w - 2 * pad - 2 * gap - barW) / across, (h - top - bottom - stack) / down));
    const mid = top + (h - top - bottom) / 2;
    const colW = Math.max(dims[0][0], dims[1][0]) * u, s1h = dims[0][1] * u, l1h = dims[1][1] * u;
    const place = [
      [pad + (colW - dims[0][0] * u) / 2, mid - (s1h + stack + l1h) / 2],
      [pad + (colW - dims[1][0] * u) / 2, mid - (s1h + stack + l1h) / 2 + s1h + stack],
      [pad + colW + gap, mid - (dims[2][1] * u) / 2],
    ];
    for (let i = 0; i < 3; i++) {
      const L = A[i], [bw, bh] = [dims[i][0] * u, dims[i][1] * u], [ox, oy] = place[i];
      const ran = i <= depth;
      const b = render(i, ran ? t : -1);
      x.imageSmoothingEnabled = false;
      x.globalAlpha = ran ? 1 : 0.35;
      x.drawImage(b.c, ox, oy, bw, bh);
      x.globalAlpha = 1;
      const what = i === 0 ? `S1 · DoG ${L.maps / 2} SCALES × ON/OFF` : `${NAMES[i]} · ${L.maps} MAPS ${L.h}×${L.w}` + (L.th ? ` · θ ${L.th}` : "");
      label(x, what, ox, oy - 8, ran ? G[3] : pa(0.4));
      M.boxes.push({ pop: i, ox, oy, u, cols: b.cols, L });
      // the neurons that learn and the square each silences
      if (r && i > 0 && i === r.depth && i < 3) {
        for (const wn of r.winners) {
          if (wn.t + 1 > t) continue;
          const [y, xx, m] = split(L, wn.unit), [tc, tr] = tileOf(i, m, b.cols);
          const cx = ox + (tc * (L.w + 1) + xx + 0.5) * u, cy = oy + (tr * (L.h + 1) + y + 0.5) * u, rad = L.radius;
          x.strokeStyle = ORANGE; x.lineWidth = 1.5; bracket(x, cx, cy, Math.max(4, 2 * u), 3);
          if (rad > 0) { x.setLineDash([3, 3]); x.strokeStyle = "rgba(236,131,90,0.7)"; x.strokeRect(cx - (rad + 0.5) * u, cy - (rad + 0.5) * u, (2 * rad + 1) * u, (2 * rad + 1) * u); x.setLineDash([]); }
        }
      }
      // probe
      if (P.probe.pop === i) {
        const [y, xx, m] = split(L, P.probe.unit), [tc, tr] = tileOf(i, m, b.cols);
        x.strokeStyle = G.p; x.lineWidth = 1.5;
        bracket(x, ox + (tc * (L.w + 1) + xx + 0.5) * u, oy + (tr * (L.h + 1) + y + 0.5) * u, Math.max(5, 2.5 * u), 3);
      }
      const k = A[i + 1].kernel, conv = (L.pool > 1 ? `pool ${L.pool} · ` : "") + `${k}×${k}`;
      if (i === 0) {
        // down from S1 into L1
        const cx = ox + bw / 2, y0 = oy + bh + 4, y1 = place[1][1] - 24;
        x.strokeStyle = G[4]; x.fillStyle = G[4]; x.lineWidth = 1.4;
        x.beginPath(); x.moveTo(cx, y0); x.lineTo(cx, y1); x.stroke();
        x.beginPath(); x.moveTo(cx, y1); x.lineTo(cx - 4, y1 - 6); x.lineTo(cx + 4, y1 - 6); x.fill();
        x.font = `600 10px ${FH}`; x.textAlign = "left"; x.fillText(conv, cx + 8, (y0 + y1) / 2 + 3);
      } else {
        const nx = i === 1 ? place[2][0] : ox + bw + gap;
        arrow(x, ox + bw + 6, nx - 6, i === 1 ? oy + bh / 2 : mid, conv);
      }
    }
    const dx = place[2][0] + dims[2][0] * u + gap;
    drawDecision(x, dx, mid, Math.min(barW, w - dx - pad), Math.min(h - top - bottom, 280), t, depth);
    // phase and its learning rates
    if (r) {
      const f = (v) => (Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(1));
      const rates = r.phase < 2 ? `a+ ${f(r.rates[2 * r.phase])}  a− ${f(-r.rates[2 * r.phase + 1])}` : `reward ${f(r.rates[4])} / ${f(r.rates[5])}  punish ${f(r.rates[6])} / ${f(r.rates[7])}`;
      const head = `PHASE ${r.phase + 1}/3 · ${PHASES[r.phase].toUpperCase()}`;
      x.font = `600 12px ${FH}`; x.textAlign = "left"; x.fillStyle = G.p;
      x.fillText(head, pad, 40);
      x.fillStyle = G[3]; x.fillText(`${rates} · CONVERGENCE ${r.conv[Math.min(r.phase, 2)].toFixed(4)}`, pad + x.measureText(head).width + 16, 40);
    }
  }

  // class evidence: the largest decision potential of each class so far, as
  // its lead over the weakest class at the end
  function drawDecision(x, ox, mid, bw, bh, t, depth) {
    const r = M.rec, C = S.meta ? S.meta.classes : 10, row = Math.min(22, bh / C), oy = mid - (row * C) / 2;
    label(x, `L3 · ${M.arch.layers[3].maps} MAPS · ${M.arch.per} / CLASS`, ox, oy - 8, depth === 3 ? G[3] : pa(0.4));
    if (!r || depth < 3 || !r.steps) {
      x.font = `13px ${FH}`; x.fillStyle = pa(0.45); x.textAlign = "left";
      x.fillText("runs once the features", ox, oy + 16); x.fillText("have learned", ox, oy + 32);
      return;
    }
    const k0 = Math.max(0, Math.min(t, r.steps) - 1), ev = r.evidence.subarray(k0 * r.classes, (k0 + 1) * r.classes);
    const end = r.evidence.subarray((r.steps - 1) * r.classes);
    // potentials are large and close; the stub below the weakest class keeps it visible
    let top = 0, lo = Infinity; for (const e of end) { top = Math.max(top, e); lo = Math.min(lo, e); }
    lo -= 0.1 * (top - lo);
    let lead = -1, best = 0; ev.forEach((e, k) => { if (e > best) { best = e; lead = k; } });
    const L0 = 18;
    M.bars = { ox, oy, row, C };
    for (let k = 0; k < C; k++) {
      const y = oy + k * row, len = top > lo ? (Math.max(0, ev[k] - lo) / (top - lo)) * (bw - L0 - 4) : 0;
      x.fillStyle = k === lead ? G.p : G[6]; x.fillRect(ox + L0, y + 3, len, row - 6);
      x.font = `${k === S.label ? 600 : 400} 13px ${FH}`; x.textAlign = "right"; x.fillStyle = k === S.label ? G.p : G[2];
      x.fillText(k, ox + L0 - 5, y + row / 2 + 4);
    }
    const done = t >= S.T;
    x.font = `600 10px ${FH}`; x.textAlign = "left"; x.fillStyle = done ? G.p : G[3];
    x.fillText(done ? "ANSWER: LARGEST FINAL POTENTIAL" : `STEP ${t}/${S.T} · LEADER ${lead < 0 ? "–" : lead}`, ox, oy + C * row + 14);
  }

  function hit(mx, my) {
    for (const b of M.boxes) {
      const L = b.L, tw = (L.w + 1) * b.u, th = (L.h + 1) * b.u;
      const tc = Math.floor((mx - b.ox) / tw), tr = Math.floor((my - b.oy) / th);
      if (tc < 0 || tr < 0) continue;
      const xx = Math.floor((mx - b.ox - tc * tw) / b.u), y = Math.floor((my - b.oy - tr * th) / b.u);
      if (xx >= L.w || y >= L.h) continue;
      const m = b.pop === 0 ? 2 * tc + tr : tr * b.cols + tc;
      if ((b.pop === 0 && tr > 1) || (b.pop > 0 && tc >= b.cols) || m >= L.maps) continue;
      return [b.pop, (y * L.w + xx) * L.maps + m];
    }
    return null;
  }

  registerView("dcsnn", popView({
    fig1: "Fig. 01 — spike wave S1 → L1 → L2 → decision · cell shade = spike time · click to probe",
    draw,
    hit(mx, my) { const u = hit(mx, my); return u && u[0] > 0 ? u : null; },
    tip(mx, my) {
      const u = hit(mx, my);
      if (!u) return null;
      const L = M.arch.layers[u[0]], [y, xx, m] = split(L, u[1]), ft = M.first[u[0]] ? M.first[u[0]][u[1]] : NONE_T;
      const what = u[0] === 0 ? `${["ON", "OFF"][m & 1]} scale ${(m >> 1) + 1}` : `map ${m}`;
      return [`${NAMES[u[0]]} · ${what} · (${y}, ${xx})`, ft === NONE_T ? " · silent" : ` · spiked at step ${ft}` + (u[0] ? " · click to probe" : "")];
    },
    H,
    pops() { M.first = S.pops.map((q) => new Uint8Array(q.size).fill(NONE_T)); },
    image() { M.first.forEach((f) => f.fill(NONE_T)); M.rec = null; },
    step(t, lists) {
      lists.forEach((a, q) => { const f = M.first[q]; if (f) for (const u of a) if (f[u] === NONE_T) f[u] = t; });
    },
    outcome(o) {
      const r = M.rec;
      if (r && r.depth < 3) o.note = `learning ${NAMES[r.depth]} features`;
    },
  }));
}
