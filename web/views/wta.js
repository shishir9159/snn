// Winner-take-all view (mnist-snn-rs): records 3 STEPS, 4 RESULT, 5 WEIGHTS, 9 THRESH.
"use strict";
{
  const probe = () => (S.pinned >= 0 ? S.pinned : S.winner >= 0 ? S.winner : Math.max(0, S.lastWinner));

  const markup = {
    firing: `
      <div class="k">Firing</div>
      <div class="meter"><div><span>Input</span><span class="num" id="in-hz">0</span></div><div class="track"><div class="fill" id="in-bar"></div></div></div>
      <div class="meter"><div><span>Output</span><span class="num" id="out-hz">0</span></div><div class="track"><div class="fill" id="out-bar"></div></div></div>
    `,
    center: `
      <div class="bp"><canvas id="net" style="cursor:crosshair"></canvas></div>
      <div class="bp"><span class="k">Weights</span><canvas id="weights" style="cursor:pointer"></canvas></div>
    `,
    right: `
      <div>
        <div class="k">Probe</div>
        <div class="num probe-name" id="probe-v">–</div>
      </div>
      <div class="bp" style="height:96px"><canvas id="trace"></canvas></div>
      <div class="pair">
        <div><span class="k">V_m</span><span class="num" id="vm-v">–</span></div>
        <div><span class="k">V_th</span><span class="num" id="vth-v">–</span></div>
        <div><span class="k">θ</span><span class="num" id="theta-v">–</span></div>
      </div>
      <div class="decision">
        <span class="k">Decision</span>
        <div class="verdict">
          <span class="num digit" id="digit-v">–</span>
          <div class="truth"><span id="truth-v"></span><span class="badge" id="badge-v"></span></div>
        </div>
        <div style="flex:1;min-height:90px"><canvas id="classes"></canvas></div>
      </div>
    `,
    bottom: `
      <canvas id="raster"></canvas>
    `,
  };

  function renderProbe() {
    const j = probe();
    const lab = S.labels[j];
    $("probe-v").textContent = `N${String(j).padStart(2, "0")}` + (lab === NONE ? "" : ` · ${lab}`);
    const p = S.pot[j];
    $("vm-v").textContent = p <= S.floor + 1 ? "inhib" : p.toFixed(1);
    $("vth-v").textContent = S.th[j].toFixed(1);
    $("theta-v").textContent = (S.theta[j] >= 0 ? "+" : "") + S.theta[j].toFixed(2);
  }

  function renderDecision() {
    const r = S.result;
    const w = r ? r.winner : S.winner;
    const pred = w >= 0 ? S.labels[w] : NONE;
    $("digit-v").textContent = pred === NONE ? "–" : pred;
    $("truth-v").textContent = S.label >= 0 ? `label ${S.label}` + (S.learn ? "" : " · test") : "";
    const b = $("badge-v");
    if (r) { b.className = "badge " + (r.correct ? "ok" : "miss"); b.textContent = r.correct ? "✓ correct" : w < 0 ? "✗ no winner" : "✗ wrong"; }
    else { b.className = "badge"; b.textContent = ""; }
  }

  let netGeom = null;
  function drawNet() {
    const g = fit($("net"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, n = S.side, N = S.nOut;
    rulers(x, w, h);
    const pad = 26, size = Math.min(h - 2 * pad, (w - 2 * pad) * 0.62), cs = size / n, ox = pad + 10, oy = (h - size) / 2;
    // two columns of output neurons when one would crowd them
    const cols = (h - 2 * pad) / N < 12 ? 2 : 1, rows = Math.ceil(N / cols), colW = 46;
    const outX = Math.min(w - 40 - colW * (cols - 1), ox + size + Math.max(90, (w - ox - size) * 0.5));
    const gap = Math.min(22, (h - 2 * pad) / rows), oy2 = h / 2 - (gap * (rows - 1)) / 2, r = Math.max(2.5, Math.min(7, gap * 0.36));
    const nodes = []; for (let j = 0; j < N; j++) nodes.push([outX + (j % cols) * colW, oy2 + gap * Math.floor(j / cols)]);
    netGeom = { nodes, r: Math.max(10, gap / 2) };

    x.font = `600 12px ${FH}`; x.fillStyle = G[3]; x.textAlign = "left";
    x.fillText(`INPUT ${n}×${n}`, ox, oy - 8);
    x.textAlign = "center"; x.fillText(`OUTPUT ×${N}`, outX + (colW * (cols - 1)) / 2, Math.max(16, oy2 - 14));

    // resting inputs: faint crosses; stroke pixels: dim fill under the spikes
    x.strokeStyle = pa(0.16); x.lineWidth = 1; x.beginPath();
    const a0 = cs * 0.12;
    for (let i = 0; i < S.nIn; i++) {
      if (S.img[i] > 20 || S.inGlow[i] > 0.05) continue;
      const cx = ox + ((i % n) + 0.5) * cs, cy = oy + (((i / n) | 0) + 0.5) * cs;
      x.moveTo(cx - a0, cy); x.lineTo(cx + a0, cy); x.moveTo(cx, cy - a0); x.lineTo(cx, cy + a0);
    }
    x.stroke();
    for (let i = 0; i < S.nIn; i++) {
      const px = S.img[i], gl = S.inGlow[i];
      const cx = ox + ((i % n) + 0.5) * cs, cy = oy + (((i / n) | 0) + 0.5) * cs;
      if (px > 20) { x.fillStyle = `rgba(17,134,92,${(px / 255) * 0.55})`; x.fillRect(cx - cs / 2 + 1, cy - cs / 2 + 1, cs - 2, cs - 2); }
      if (gl > 0.05) {
        const s = (cs - 2) * Math.min(1, 0.45 + gl * 0.6);
        x.fillStyle = gb(Math.min(1, gl * 1.2)); x.fillRect(cx - s / 2, cy - s / 2, s, s);
        if (gl > 0.6) { const c = s * 0.36; x.fillStyle = pa(gl); x.fillRect(cx - c / 2, cy - c / 2, c, c); }
      }
    }

    // synapses from recently active inputs to the probed neuron, weighted
    const j0 = probe(), [tx, ty] = nodes[j0];
    if (S.weights) {
      const row = j0 * S.nIn;
      x.lineWidth = 1;
      for (let i = 0; i < S.nIn; i++) {
        const gl = S.inGlow[i];
        if (gl < 0.08) continue;
        const wv = S.weights[row + i] / 255;
        if (wv < 0.05) continue;
        x.strokeStyle = ga(gl * wv * 0.9); x.beginPath();
        x.moveTo(ox + ((i % n) + 0.5) * cs, oy + (((i / n) | 0) + 0.5) * cs); x.lineTo(tx - r - 2, ty); x.stroke();
      }
    }

    const floor = S.floor + 1;
    for (let j = 0; j < N; j++) {
      const [cx, cy] = nodes[j], gl = S.outGlow[j], p = S.pot[j];
      if (gl > 0.03) {
        const rr = r + 18 * gl, gr = x.createRadialGradient(cx, cy, 0, cx, cy, rr);
        gr.addColorStop(0, gb(0.6 * gl)); gr.addColorStop(1, gb(0));
        x.fillStyle = gr; x.beginPath(); x.arc(cx, cy, rr, 0, 7); x.fill();
      }
      x.beginPath(); x.arc(cx, cy, r, 0, 7); x.fillStyle = G[9]; x.fill();
      x.strokeStyle = p <= floor ? pa(0.15) : pa(0.45); x.lineWidth = 1; x.stroke();
      const v = Math.max(0, Math.min(1, p / S.th[j]));
      if (v > 0.03) { x.beginPath(); x.arc(cx, cy, r * v, 0, 7); x.fillStyle = G[5]; x.fill(); }
      if (gl > 0.03) { x.beginPath(); x.arc(cx, cy, r, 0, 7); x.fillStyle = pa(gl); x.fill(); }
      if (S.labels[j] !== NONE && gap >= 9) {
        x.font = `500 ${Math.min(13, gap * 0.8)}px ${FH}`; x.fillStyle = j === S.winner ? G.p : pa(0.5); x.textAlign = "left";
        x.fillText(S.labels[j], cx + r + 8, cy + 4);
      }
    }
    if (S.winner >= 0) { const [cx, cy] = nodes[S.winner]; x.strokeStyle = ga(1); x.lineWidth = 1; bracket(x, cx, cy, r + 5, 3); }
    x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, tx, ty, r + 9, 4);
    if (S.pinned >= 0) {
      x.setLineDash([4, 4]); x.strokeStyle = gb(0.5); x.lineWidth = 1;
      x.beginPath(); x.moveTo(tx + r + 12, Math.round(ty) + 0.5); x.lineTo(w, Math.round(ty) + 0.5); x.stroke(); x.setLineDash([]);
    }
  }

  let weightTiles = null, weightGeom = null;
  const RAMP = (() => { // weight byte → phthalo sequential ramp, dark (low) to pale (high)
    const stops = [[0, 8, 26, 18], [0.35, 12, 74, 51], [0.6, 17, 134, 92], [0.8, 82, 196, 149], [1, 238, 242, 238]];
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255; let k = 1; while (stops[k][0] < t) k++;
      const [t0, ...c0] = stops[k - 1], [t1, ...c1] = stops[k], f = (t - t0) / (t1 - t0);
      for (let c = 0; c < 3; c++) out[i * 3 + c] = c0[c] + (c1[c] - c0[c]) * f;
    }
    return out;
  })();

  function buildTiles() {
    const n = S.side, N = S.nOut, c = document.createElement("canvas");
    c.width = n * N; c.height = n;
    const x = c.getContext("2d"), im = x.createImageData(n * N, n);
    for (let j = 0; j < N; j++) for (let i = 0; i < S.nIn; i++) {
      const b = S.weights[j * S.nIn + i], px = ((i / n) | 0) * n * N + j * n + (i % n);
      im.data[px * 4] = RAMP[b * 3]; im.data[px * 4 + 1] = RAMP[b * 3 + 1]; im.data[px * 4 + 2] = RAMP[b * 3 + 2]; im.data[px * 4 + 3] = 255;
    }
    x.putImageData(im, 0, 0);
    return c;
  }

  function drawWeights() {
    const g = fit($("weights"));
    if (!g || !S.weights) return;
    const { x, w, h } = g, N = S.nOut, n = S.side, top = 30, pad = 12;
    if (!weightTiles) weightTiles = buildTiles();
    let best = { s: 0 };
    for (let cols = 1; cols <= N; cols++) {
      const rows = Math.ceil(N / cols), s = Math.min((w - 2 * pad) / cols, (h - top - pad) / rows);
      if (s > best.s) best = { s, cols, rows };
    }
    const { s, cols } = best, gapPx = Math.max(3, s * 0.1), t = s - gapPx;
    const ox = (w - cols * s) / 2;
    weightGeom = { ox, top, s, t, cols };
    x.imageSmoothingEnabled = false;
    for (let j = 0; j < N; j++) {
      const tx = ox + (j % cols) * s, ty = top + Math.floor(j / cols) * s;
      x.drawImage(weightTiles, j * n, 0, n, n, tx, ty, t, t);
      if (S.labels[j] !== NONE && t >= 26) {
        x.font = `600 ${Math.min(14, t * 0.22)}px ${FH}`; x.textAlign = "left"; x.fillStyle = pa(0.75);
        x.fillText(S.labels[j], tx + 3, ty + Math.min(14, t * 0.22) + 1);
      }
      if (j === S.winner || (S.result && j === S.result.winner)) { x.strokeStyle = ga(1); x.lineWidth = 1; bracket(x, tx + t / 2, ty + t / 2, t / 2 + 2, 4); }
      if (j === probe()) { x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, tx + t / 2, ty + t / 2, t / 2 + 3, 6); }
    }
  }

  function drawTrace() {
    const g = fit($("trace"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, j = probe(), th = S.th[j], T = S.T;
    const lo = -0.25 * th, hi = 1.15 * th, Y = (v) => h - 6 - ((Math.max(lo, v) - lo) / (hi - lo)) * (h - 18), X = (t) => ((t - 1) / (T - 1)) * w;
    x.fillStyle = pa(0.06); for (let k = 1; k < 4; k++) x.fillRect(0, Math.round((h * k) / 4), w, 1);
    x.setLineDash([4, 4]); x.strokeStyle = pa(0.5); x.lineWidth = 1;
    x.beginPath(); x.moveTo(0, Y(th)); x.lineTo(w, Y(th)); x.stroke(); x.setLineDash([]);
    x.fillStyle = G.p; x.font = `600 10px ${FH}`; x.textAlign = "right"; x.fillText("V_TH", w - 6, Y(th) - 4);
    const floor = S.floor + 1;
    x.strokeStyle = G[3]; x.lineWidth = 1.5; x.beginPath();
    for (let t = 1; t <= S.t; t++) {
      const v = S.trace[t * S.nOut + j], px = X(t), spiked = S.spk[t * S.nOut + j];
      if (v <= floor) { x.fillStyle = pa(0.05); x.fillRect(px, 0, w / T + 0.5, h); }
      if (t === 1) x.moveTo(px, Y(v)); else x.lineTo(px, Y(spiked ? th : v));
      if (spiked) { x.stroke(); x.strokeStyle = G.p; x.beginPath(); x.moveTo(px, Y(th)); x.lineTo(px, 3); x.stroke(); x.strokeStyle = G[3]; x.beginPath(); x.moveTo(px, Y(v)); }
    }
    x.stroke();
  }

  function classEvidence() {
    const C = S.meta.classes, ev = new Float32Array(C);
    for (let j = 0; j < S.nOut; j++) { const c = S.labels[j]; if (c < C) ev[c] = Math.max(ev[c], Math.min(1.25, S.peak[j])); }
    return ev;
  }

  let classGeom = null;
  function drawClasses() {
    const g = fit($("classes"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, C = S.meta.classes, ev = classEvidence(), band = w / C, bw = Math.min(24, band - 6), base = h - 18;
    const r = S.result, pred = (r ? r.winner : S.winner) >= 0 ? S.labels[r ? r.winner : S.winner] : NONE;
    classGeom = { band, ev };
    x.fillStyle = pa(0.18); x.fillRect(0, base, w, 1);
    const y1 = base - (1 / 1.25) * (base - 4);
    x.setLineDash([3, 3]); x.strokeStyle = pa(0.3); x.beginPath(); x.moveTo(0, y1); x.lineTo(w, y1); x.stroke(); x.setLineDash([]);
    for (let c = 0; c < C; c++) {
      const cx = band * c + band / 2, bh = (ev[c] / 1.25) * (base - 4);
      if (bh > 0.5) {
        x.fillStyle = c === pred ? G[2] : G[6];
        const rr = Math.min(4, bh, bw / 2);
        x.beginPath(); x.moveTo(cx - bw / 2, base); x.lineTo(cx - bw / 2, base - bh + rr); x.arcTo(cx - bw / 2, base - bh, cx, base - bh, rr);
        x.arcTo(cx + bw / 2, base - bh, cx + bw / 2, base - bh + rr, rr); x.lineTo(cx + bw / 2, base); x.closePath(); x.fill();
      }
      x.font = `${c === S.label ? 600 : 400} 13px ${FH}`; x.textAlign = "center"; x.fillStyle = c === S.label ? G.p : G[2];
      x.fillText(c, cx, h - 3);
    }
  }

  function drawRaster() {
    const g = fit($("raster"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, L = S.ring.length, L0 = 46, inH = (h - 16) * 0.68, outTop = inH + 10, outH = h - outTop - 6;
    const cw = (w - L0 - 8) / L, rIn = inH / S.nIn, rOut = outH / S.nOut;
    x.font = `600 11px ${FH}`; x.textAlign = "left";
    x.fillStyle = G[3]; x.fillText("INPUT", 8, inH / 2 + 4); x.fillStyle = G.p; x.fillText("OUTPUT", 8, outTop + outH / 2 + 4);
    x.fillStyle = pa(0.12); x.fillRect(L0, outTop - 5, w - L0, 1);
    for (let k = 0; k < L; k++) {
      const e = S.ring[(S.ringAt + k) % L];
      if (!e) continue;
      const px = L0 + k * cw;
      if (e.mark !== undefined) {
        x.fillStyle = pa(0.25); x.fillRect(Math.round(px) + 0.5, 0, 1, h);
        x.fillStyle = G[2]; x.fillText(e.mark, px + 4, 11); continue;
      }
      x.fillStyle = ga(0.85);
      for (const i of e.a) x.fillRect(px, 4 + i * rIn, Math.max(1.2, cw), Math.max(1, rIn));
      x.fillStyle = G.p;
      for (const j of e.f) x.fillRect(px - 0.5, outTop + j * rOut, Math.max(2, cw + 1), Math.max(1.5, rOut - 0.5));
    }
    x.fillStyle = G.p; x.fillRect(w - 8, 0, 1, h);
  }

  function bind() {
    const neuronText = (j) => `N${String(j).padStart(2, "0")} · label ${S.labels[j] === NONE ? "–" : S.labels[j]} · θ ${S.theta[j].toFixed(2)}`;
    const pin = (j) => { S.pinned = S.pinned === j ? -1 : j; dirty("all"); };

    function netHit(e) {
      if (!netGeom) return -1;
      const [mx, my] = local(e);
      let best = -1, bd = netGeom.r * netGeom.r;
      netGeom.nodes.forEach(([px, py], j) => { const d = (px - mx) ** 2 + (py - my) ** 2; if (d < bd) { bd = d; best = j; } });
      return best;
    }
    $("net").onpointermove = (e) => { const j = netHit(e); j < 0 ? hideTip() : showTip(e, `${(S.pot[j] / S.th[j] * 100).toFixed(0)}%`, `of threshold · ${neuronText(j)}`); };
    $("net").onpointerleave = hideTip;
    $("net").onclick = (e) => { const j = netHit(e); if (j >= 0) pin(j); };

    function weightHit(e) {
      if (!weightGeom) return -1;
      const [mx, my] = local(e), { ox, top, s, t, cols } = weightGeom;
      const c = Math.floor((mx - ox) / s), r = Math.floor((my - top) / s), j = r * cols + c;
      return c >= 0 && c < cols && r >= 0 && j < S.nOut && (mx - ox) % s <= t ? j : -1;
    }
    $("weights").onpointermove = (e) => { const j = weightHit(e); j < 0 ? hideTip() : showTip(e, `N${String(j).padStart(2, "0")}`, `label ${S.labels[j] === NONE ? "–" : S.labels[j]} · θ ${S.theta[j].toFixed(2)}`); };
    $("weights").onpointerleave = hideTip;
    $("weights").onclick = (e) => { const j = weightHit(e); if (j >= 0) pin(j); };

    $("classes").onpointermove = (e) => {
      if (!classGeom) return;
      const c = Math.floor(local(e)[0] / classGeom.band);
      if (c < 0 || c >= S.meta.classes) return hideTip();
      showTip(e, `${(classGeom.ev[c] * 100).toFixed(0)}%`, `of threshold · best class ${c} neuron`);
    };
    $("classes").onpointerleave = hideTip;
    addEventListener("keydown", (e) => { if (e.key === "Escape" && S.pinned >= 0) pin(S.pinned); });
  }

  function stats() {
    const t = Math.max(1, S.t);
    const inHz = (S.inSpikes / S.nIn) / t * 1000, outHz = (S.outSpikes / S.nOut) / t * 1000;
    $("in-hz").textContent = inHz.toFixed(1); $("out-hz").textContent = outHz.toFixed(1);
    $("in-bar").style.width = Math.min(100, inHz / 40 * 100) + "%";
    $("out-bar").style.width = Math.min(100, outHz / 40 * 100) + "%";
  }

  registerView("wta", {
    layout() {
      for (const [id, html] of Object.entries(markup)) $(id).innerHTML = html;
      bind();
    },
    pops() {
      S.t = 0; S.winner = -1; S.lastWinner = -1; S.pinned = -1; S.inSpikes = 0; S.outSpikes = 0; S.weights = null; S.result = null;
      const out = S.pops.find((q) => q.role === 5), N = out ? out.size : 0, T = S.T;
      S.nOut = N;
      S.th = new Float32Array(N).fill(1); S.pot = new Float32Array(N); S.peak = new Float32Array(N);
      S.trace = new Float32Array((T + 1) * N); S.spk = new Uint8Array((T + 1) * N); S.inGlow = new Float32Array(S.nIn); S.outGlow = new Float32Array(N);
      S.labels = new Uint8Array(N).fill(NONE); S.theta = new Float32Array(N);
      S.ring = new Array(T * 3).fill(null); S.ringAt = 0;
      dirty("all");
    },
    image() {
      S.trace.fill(0); S.spk.fill(0); S.peak.fill(0); S.t = 0; S.winner = -1; S.result = null; S.inSpikes = 0; S.outSpikes = 0;
      S.ring[S.ringAt] = { mark: S.label }; // boundary marker occupies one slot
      S.ringAt = (S.ringAt + 1) % S.ring.length;
      dirty("net", "trace", "decision", "classes");
    },
    H: {
    9(v, p) { for (let j = 0; j < S.nOut; j++) S.th[j] = v.getFloat32(p + 4 * j, true); }, // THRESH
    3(v, p) { // STEPS
      let t = v.getUint16(p, true), o = p + 4;
      const n = v.getUint16(p + 2, true);
      for (let k = 0; k < n; k++, t++) {
        const na = v.getUint16(o, true); o += 2;
        const a = new Uint16Array(na);
        for (let q = 0; q < na; q++, o += 2) { a[q] = v.getUint16(o, true); S.inGlow[a[q]] = 1; }
        const nf = v.getUint16(o, true); o += 2;
        const f = new Uint16Array(nf);
        for (let q = 0; q < nf; q++, o += 2) { f[q] = v.getUint16(o, true); S.outGlow[f[q]] = 1; S.spk[t * S.nOut + f[q]] = 1; }
        for (let j = 0; j < S.nOut; j++, o += 4) {
          const pv = v.getFloat32(o, true);
          S.pot[j] = pv; S.trace[t * S.nOut + j] = pv;
          const r = pv / S.th[j];
          if (r > S.peak[j]) S.peak[j] = r;
        }
        S.winner = v.getInt16(o, true); o += 2;
        if (S.winner >= 0) S.lastWinner = S.winner;
        S.inSpikes += na; S.outSpikes += nf; S.t = t;
        S.ring[S.ringAt] = { a, f };
        S.ringAt = (S.ringAt + 1) % S.ring.length;
      }
      dirty("net", "trace", "raster", "classes", "decision");
    },
    4(v, p) { // RESULT
      const w = v.getInt16(p + 4, true), n = S.nOut;
      S.result = { winner: w, predicted: v.getUint8(p + 6), correct: v.getUint8(p + 7) === 1, learn: v.getUint8(p + 8) === 1 };
      let o = p + 13 + 2 * n;
      for (let j = 0; j < n; j++, o += 4) S.theta[j] = v.getFloat32(o, true);
      for (let j = 0; j < n; j++, o++) S.labels[j] = v.getUint8(o);
      dirty("decision", "weights", "net");
    },
    5(v, p) { // WEIGHTS
      const rows = v.getUint16(p, true), cols = v.getUint16(p + 2, true);
      let o = p + 4 + rows * cols;
      S.weights = new Uint8Array(v.buffer.slice(v.byteOffset + p + 4, v.byteOffset + o));
      for (let j = 0; j < rows; j++, o++) S.labels[j] = v.getUint8(o);
      for (let j = 0; j < rows; j++, o += 4) S.theta[j] = v.getFloat32(o, true);
      weightTiles = null; dirty("weights", "net");
    },
    },
    stats,
    frame(all) {
      let glowing = false;
      for (let i = 0; i < S.nIn; i++) if (S.inGlow[i] > 0.01) { S.inGlow[i] *= 0.88; glowing = true; }
      for (let j = 0; j < S.nOut; j++) if (S.outGlow[j] > 0.01) { S.outGlow[j] *= 0.9; glowing = true; }
      if (glowing) dirty("net");
      if (all || S.dirty.has("net")) drawNet();
      if (all || S.dirty.has("weights")) drawWeights();
      if (all || S.dirty.has("trace")) { drawTrace(); renderProbe(); }
      if (all || S.dirty.has("classes")) drawClasses();
      if (all || S.dirty.has("decision")) renderDecision();
      if (all || S.dirty.has("raster")) drawRaster();
    },
  });
}
