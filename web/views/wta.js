// Winner-take-all view (mnist-snn-rs): records 3 STEPS, 4 RESULT, 5 WEIGHTS, 9 THRESH,
// 14 DOPAMINE, 15 TAGS under the delayed reward rule and 51 VOLTAGE under the voltage rule.
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
    // rows still holding tags, brighter the fresher
    if (DA.on && DA.rec) {
      for (const [j, tg] of DA.tags) {
        if (j >= N) continue;
        const tx = ox + (j % cols) * s, ty = top + Math.floor(j / cols) * s, a = Math.exp(-tg.age / DA.rec.tauC);
        x.beginPath(); x.arc(tx + t - 5, ty + 5, Math.max(2, Math.min(4, t * 0.06)), 0, 7); x.fillStyle = `rgba(238,242,238,${0.25 + 0.75 * a})`; x.fill();
      }
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

  // ---- delayed reward: records 14 DOPAMINE and 15 TAGS ---------------------------------------
  // Each answer tags its winner's synapses; the reward lands later as dopamine and
  // changes whatever is still tagged. The timeline is rebuilt from the recent trials:
  // dopamine is a sum of decaying puffs, each trial's tags fade with τc.
  const DA = { on: false, rec: null, shown: 0, last: 0, tags: new Map(), geom: null, hover: null };

  function enableDopamine() {
    DA.on = true;
    const narrow = matchMedia("(max-width: 1100px)").matches;
    document.querySelector(".app").style.gridTemplateRows = narrow ? "auto auto 230px" : "auto minmax(0, 1fr) 230px";
    const cell = (id, more = "") => `<div style="position:relative;min-width:0${more}"><canvas id="${id}" style="position:absolute;inset:0"></canvas></div>`;
    $("bottom").innerHTML = `
      <div style="display:grid;grid-template-columns:minmax(0,1fr) 180px minmax(0,2.6fr);grid-template-rows:minmax(0,1fr);height:100%">
        ${cell("raster")}${cell("tags", ";border-left:1px solid var(--divider)")}${cell("dopa", ";border-left:1px solid var(--divider)")}
      </div>`;
    $("dopa").style.cursor = "crosshair";
    corners(document);
    $("dopa").onpointermove = (e) => {
      const t = trialAt(local(e)[0]);
      if (t !== DA.hover) { DA.hover = t; dirty("dopa"); }
      if (!t) return hideTip();
      const sign = t.amount > 0 ? "reward" : "punishment", late = ((t.at - t.start - t.decided) / 1000).toFixed(2);
      const answer = t.winner < 0 ? "no answer" : `N${String(t.winner).padStart(2, "0")} answered ${t.winner % S.meta.classes}`;
      const fate = t.winner < 0 ? "" : Number.isNaN(t.credit) ? ` · ${sign} ${fmtAmount(t.amount)} on its way, ${late} s late`
        : ` · ${sign} ${fmtAmount(t.amount)} landed ${late} s late; N${String(t.winner).padStart(2, "0")} held ${(t.credit * 100).toFixed(0)}% of the tags then`;
      showTip(e, `label ${t.label}`, ` · ${answer}${fate}`);
    };
    $("dopa").onpointerleave = () => { DA.hover = null; dirty("dopa"); hideTip(); };
    dirty("all");
  }

  const fmtAmount = (a) => (a > 0 ? "+" : "−") + Math.abs(a).toFixed(2);

  // training time on screen: the last trial's gap plays out at the run's pace, then
  // the next training image's steps carry on from it
  function daNow() {
    const r = DA.rec;
    if (!r) return 0;
    return DA.shown < r.clock ? DA.shown : r.clock + (S.learn ? S.t : 0);
  }

  function daTick() {
    const r = DA.rec, now = performance.now(), dt = Math.min(100, now - DA.last);
    DA.last = now;
    if (!r || DA.shown >= r.clock) return false;
    const sp = S.state.speed;
    if (!sp) DA.shown = r.clock;
    else if (!S.state.paused) DA.shown = Math.min(r.clock, DA.shown + (dt * sp) / 1000);
    return true;
  }

  // dopamine at training time t from the puffs that arrived by then
  function dopamineAt(t, r, upTo) {
    let d = 0;
    for (const q of r.trials) {
      if (q.winner < 0 || q.at > t || q.at > upTo) continue;
      d += r.pulse * q.amount * Math.pow(r.kd, t - q.at);
    }
    return d;
  }

  // a trial's tags at t: growing over its steps from the answer, then fading
  function tagAt(q, t, r) {
    const t0 = q.start + q.decided, t1 = q.start + r.steps;
    if (q.winner < 0 || t < t0) return 0;
    if (t <= t1) return t1 > t0 ? (q.tagged * (t - t0)) / (t1 - t0) : q.tagged;
    return q.tagged * Math.exp(-(t - t1) / r.tauC);
  }

  function trialAt(mx) {
    const gm = DA.geom;
    if (!gm) return null;
    const t = gm.t0 + ((mx - gm.L0) / gm.span) * gm.W;
    let best = null;
    for (const q of DA.rec.trials) if (q.start <= t && t <= q.start + gm.period) best = q;
    return best;
  }

  function drawDopa() {
    const g = fit($("dopa"));
    if (!g) return;
    const { x, w, h } = g, r = DA.rec, L0 = 74, R0 = 12;
    x.font = `600 11px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    if (!r) { x.fillText("DELAYED REWARD · WAITING FOR THE FIRST ANSWER", 10, 16); DA.geom = null; return; }
    const period = r.steps + r.gap, W = Math.min(20000, Math.max(3000, 4 * period, r.delay + r.jitter + 2 * r.steps));
    const now = daNow(), t0 = now - 0.8 * W, span = w - L0 - R0, X = (t) => L0 + ((t - t0) / W) * span;
    DA.geom = { t0, W, L0, span, period };
    const arcY = 52, stripY = 54, stripH = 16, daY = 78, rest = h - daY - 18, daH = rest * 0.48, tagY = daY + daH + 8, tagH = rest - daH - 8;
    x.fillText(`REWARD ${fmtS(r.delay)}–${fmtS(r.delay + r.jitter)} LATE · TAGS τ ${fmtS(r.tauC)} · DOPAMINE τ ${fmtS(r.tauD)} · GAP ${fmtS(r.gap)}`, 10, 16);
    const arrived = r.trials.filter((q) => q.winner >= 0 && !Number.isNaN(q.credit)).slice(-20);
    const credit = arrived.length ? arrived.reduce((a, q) => a + q.credit, 0) / arrived.length : NaN;
    const inFlight = r.trials.filter((q) => q.winner >= 0 && q.at > now).length;
    x.textAlign = "right"; x.fillStyle = G[2];
    x.fillText(`IN FLIGHT ${inFlight} · TAGGED ROWS ${r.live}` + (Number.isNaN(credit) ? "" : ` · CREDIT TO THE ANSWERING ROW ${(credit * 100).toFixed(0)}%`), w - R0, 16);
    x.textAlign = "left"; x.fillStyle = G[3];
    for (const [text, y] of [["REWARDS", arcY - 12], ["IMAGES", stripY + 11], ["DOPAMINE", daY + daH / 2 + 4], ["TAGS", tagY + tagH / 2 + 4]]) x.fillText(text, 10, y);
    // the future: rewards already on their way land here
    const xn = X(now);
    x.fillStyle = pa(0.04); x.fillRect(xn, 24, w - R0 - xn, h - 24 - 14);
    x.setLineDash([3, 3]); x.strokeStyle = pa(0.45); x.lineWidth = 1;
    x.beginPath(); x.moveTo(Math.round(xn) + 0.5, 24); x.lineTo(Math.round(xn) + 0.5, h - 14); x.stroke(); x.setLineDash([]);
    x.font = `600 10px ${FH}`; x.fillStyle = G[2]; x.fillText("NOW", xn + 4, h - 4);
    // axis: seconds before now
    x.textAlign = "center"; x.fillStyle = pa(0.5);
    const tick = W > 12000 ? 2000 : 1000;
    for (let m = tick; now - m > t0; m += tick) { const px = X(now - m); if (px > L0 + 10) x.fillText(`−${fmtS(m)}`, px, h - 4); }
    x.save(); x.beginPath(); x.rect(L0, 0, span, h); x.clip();

    // images: shaded by the answer, a tick where it came
    for (const q of r.trials) {
      const mid = X(q.start + r.steps / 2), half = Math.max(7, (X(q.start + r.steps) - X(q.start)) / 2);
      if (mid + half < L0 || q.start > now) continue;
      const hov = q === DA.hover;
      x.fillStyle = q.winner < 0 ? pa(hov ? 0.35 : 0.2) : q.correct ? gb(hov ? 1 : 0.7) : `rgba(236,131,90,${hov ? 1 : 0.7})`;
      x.fillRect(mid - half, stripY, 2 * half, stripH);
      x.font = `600 12px ${FH}`; x.textAlign = "center"; x.fillStyle = G[9]; x.fillText(q.label, mid, stripY + 12);
      if (q.winner >= 0) { x.fillStyle = G.p; x.fillRect(X(q.start + q.decided) - 0.5, stripY - 3, 1.5, stripH + 3); }
    }

    // dopamine: puffs that arrived, then the ones on their way, dashed
    const step = 2, n = Math.ceil(span / step) + 1, ts = new Float64Array(n), ds = new Float64Array(n);
    let top = 1e-9;
    for (let i = 0; i < n; i++) { ts[i] = t0 + ((i * step) / span) * W; ds[i] = dopamineAt(ts[i], r, Infinity); top = Math.max(top, Math.abs(ds[i])); }
    const y0 = daY + daH / 2, Y = (d) => y0 - (d / top) * (daH / 2 - 2);
    x.fillStyle = pa(0.15); x.fillRect(L0, Math.round(y0), span, 1);
    for (const [sign, color] of [[1, gb(0.75)], [-1, "rgba(236,131,90,0.8)"]]) {
      x.beginPath(); x.moveTo(L0, y0);
      for (let i = 0; i < n && ts[i] <= now; i++) x.lineTo(L0 + i * step, Y(sign * ds[i] > 0 ? ds[i] : 0));
      x.lineTo(Math.min(xn, L0 + (n - 1) * step), y0); x.closePath(); x.fillStyle = color; x.fill();
    }
    x.setLineDash([2, 3]); x.strokeStyle = pa(0.5); x.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) { if (ts[i] < now) continue; const px = L0 + i * step, py = Y(ds[i]); started ? x.lineTo(px, py) : x.moveTo(px, py); started = true; }
    x.stroke(); x.setLineDash([]);

    // tags, one band per trial stacked oldest first; brighter for the hovered one
    const live = r.trials.filter((q) => q.winner >= 0 && q.tagged > 0 && q.start + r.steps + 7 * r.tauC > t0 && q.start <= now);
    const stack = new Float64Array(n * (live.length + 1));
    let peak = 1e-9;
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = 0; k < live.length; k++) { acc += tagAt(live[k], ts[i], r); stack[i * (live.length + 1) + k + 1] = acc; }
      peak = Math.max(peak, acc);
    }
    const YT = (v) => tagY + tagH - (v / peak) * (tagH - 2);
    for (let k = 0; k < live.length; k++) {
      const q = live[k], m = live.length + 1;
      x.beginPath();
      for (let i = 0; i < n; i++) { const px = L0 + i * step, py = YT(stack[i * m + k + 1]); i ? x.lineTo(px, py) : x.moveTo(px, py); }
      for (let i = n - 1; i >= 0; i--) x.lineTo(L0 + i * step, YT(stack[i * m + k]));
      x.closePath();
      x.fillStyle = q === DA.hover ? G[2] : !DA.hover ? (k % 2 ? G[6] : G[7]) : k % 2 ? "rgba(17,134,92,0.45)" : "rgba(11,102,70,0.45)";
      x.fill();
    }
    // ahead of now the tags only fade, unless a new answer tags more
    x.fillStyle = "rgba(18,53,36,0.6)"; x.fillRect(xn, tagY, w - xn, tagH);

    // each reward: an arc from its answer to where it lands, with the share of the
    // tags its row held then
    let lastLabel = -Infinity;
    for (const q of r.trials) {
      if (q.winner < 0) continue;
      const a = X(q.start + q.decided), b = X(q.at);
      if (b < L0 - 40 || a > w) continue;
      const pending = q.at > now, hov = q === DA.hover, rise = Math.min(30, 8 + (b - a) * 0.25);
      x.strokeStyle = q.amount > 0 ? (hov ? G.p : G[4]) : ORANGE;
      x.globalAlpha = hov || !DA.hover ? 1 : 0.35;
      x.lineWidth = 1 + 1.6 * Math.min(1, Math.abs(q.amount));
      if (pending) x.setLineDash([3, 3]);
      x.beginPath(); x.moveTo(a, stripY - 2); x.quadraticCurveTo((a + b) / 2, stripY - 2 - 2 * rise, b, stripY - 2); x.stroke();
      x.setLineDash([]);
      // where it lands: on the dopamine lane, and through the tags
      x.beginPath(); x.moveTo(b, y0 - 5); x.lineTo(b + 4, y0); x.lineTo(b, y0 + 5); x.lineTo(b - 4, y0); x.closePath();
      if (pending) { x.lineWidth = 1; x.stroke(); } else { x.fillStyle = x.strokeStyle; x.fill(); }
      if (!pending) {
        x.globalAlpha *= 0.5; x.lineWidth = 1; x.setLineDash([2, 2]);
        x.beginPath(); x.moveTo(Math.round(b) + 0.5, tagY); x.lineTo(Math.round(b) + 0.5, tagY + tagH); x.stroke(); x.setLineDash([]);
        x.globalAlpha = hov || !DA.hover ? 1 : 0.35;
        if (b - lastLabel > 30 && b < xn - 4) {
          x.font = `600 10px ${FH}`; x.textAlign = "center"; x.fillStyle = G[2];
          x.fillText(`${(q.credit * 100).toFixed(0)}%`, b, tagY - 1);
          lastLabel = b;
        }
      }
      x.globalAlpha = 1;
    }
    x.restore();
  }

  const fmtS = (ms) => (ms >= 1000 ? `${+(ms / 1000).toFixed(2)} S` : `${Math.round(ms)} MS`);

  // a pinned neuron's tags, else the freshest: what a reward would do to each input now
  function tagRow() {
    if (S.pinned >= 0) return S.pinned;
    let best = probe(), age = Infinity;
    for (const [j, t] of DA.tags) if (t.age < age) { age = t.age; best = j; }
    return best;
  }

  function drawTags() {
    const g = fit($("tags"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, j = tagRow(), t = DA.tags.get(j), n = S.side;
    x.font = `600 11px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    x.fillText(`TAGS · N${String(j).padStart(2, "0")}`, 10, 16);
    const size = Math.min(w - 20, h - 62), ox = (w - size) / 2, oy = 24;
    x.strokeStyle = pa(0.2); x.lineWidth = 1; x.strokeRect(ox - 0.5, oy - 0.5, size + 1, size + 1);
    x.font = `12px ${FH}`; x.fillStyle = G[2]; x.textAlign = "left";
    if (!t || !DA.rec) { x.fillStyle = pa(0.45); x.fillText("holds no tags", ox + 6, oy + 18); return; }
    // the record is as of the end of the gap; the timeline may still be in it
    const age = Math.max(0, t.age - (DA.rec.clock - daNow())), left = Math.exp(-age / DA.rec.tauC), k = Math.sqrt(left);
    drawGrid(x, ox, oy, size, n, n, (i) => { const b = 3 * (Math.round(t.data[i] * k) + 128); return [DIVERGE[b], DIVERGE[b + 1], DIVERGE[b + 2]]; });
    x.fillText(`${(left * 100).toFixed(0)}% of a fresh tag · ${fmtS(age).toLowerCase()} old`, ox, oy + size + 15);
    x.fillStyle = pa(0.55); x.fillText("a reward grows pale, shrinks orange", ox, oy + size + 30);
  }

  // ---- voltage-based STDP: record 51 VOLTAGE -------------------------------------------------
  // The rule reads each neuron's membrane voltage u, in units of its threshold. A pre spike
  // depresses its synapse by how far ū− (a trace of u) sits above θ−; a spike potentiates each
  // input by its trace x̄ times how depolarised the neuron was just before, ū+ above θ−. The
  // traces are rebuilt here from the streamed potentials as voltage.rs builds them; the weight
  // changes come from the record, sent once a training image is over.
  const VS = { on: false, rec: null, images: 0, at: -1, winner: -1, label: -1, geom: null, bars: null, hover: -1 };
  const nName = (j) => `N${String(j).padStart(2, "0")}`;
  const fmtW = (v) => (Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(3) : v.toExponential(1));

  function enableVoltage() {
    VS.on = true;
    const narrow = matchMedia("(max-width: 1100px)").matches;
    document.querySelector(".app").style.gridTemplateRows = narrow ? "auto auto 230px" : "auto minmax(0, 1fr) 230px";
    const cell = (id, more = "") => `<div style="position:relative;min-width:0${more}"><canvas id="${id}" style="position:absolute;inset:0"></canvas></div>`;
    const line = ";border-left:1px solid var(--divider)";
    $("bottom").innerHTML = `
      <div style="display:grid;grid-template-columns:minmax(0,1fr) 180px minmax(0,2.4fr) 220px;grid-template-rows:minmax(0,1fr);height:100%">
        ${cell("raster")}${cell("vdw", line)}${cell("volt", line)}${cell("vbars", line)}
      </div>`;
    $("volt").style.cursor = "crosshair";
    $("vbars").style.cursor = "pointer";
    corners(document);
    $("volt").onpointermove = (e) => {
      const gm = VS.geom;
      if (!gm) return;
      const t = Math.round(((local(e)[0] - gm.L0) / gm.span) * gm.T + 0.5);
      if (t < 1 || t > S.t) { VS.hover = -1; dirty("volt"); return hideTip(); }
      if (t !== VS.hover) { VS.hover = t; dirty("volt"); }
      const v = gm.tr, f = (a) => a.toFixed(2);
      const plast = (v.ltd[t] > 0 ? ` · LTD ${fmtW(v.ltd[t])} over ${S.nAct[t]} inputs` : "") + (v.ltp[t] > 0 ? ` · LTP ${fmtW(v.ltp[t])}` : "");
      showTip(e, `${t} ms`, ` · u ${v.u[t] >= VS.rec.vSpike ? "spike" : f(v.u[t])} · ū− ${f(v.um[t])} · ū+ ${f(v.up[t])}${plast}`);
    };
    $("volt").onpointerleave = () => { VS.hover = -1; dirty("volt"); hideTip(); };
    const barAt = (e) => {
      const b = VS.bars;
      if (!b) return -1;
      const j = Math.floor((local(e)[0] - b.L) / b.bw);
      return j >= 0 && j < b.N ? j : -1;
    };
    $("vbars").onpointermove = (e) => {
      const j = barAt(e), r = VS.rec;
      if (j < 0) return hideTip();
      const home = r.uRef > 0 ? ` · ū̄ ${r.bar[j].toFixed(3)}, LTD ×${(r.aLtd[j] / r.aLtdBase).toFixed(2)}` : "";
      showTip(e, nName(j), ` · LTP +${fmtW(r.ltp[j])} · LTD −${fmtW(r.ltd[j])}${j === VS.winner ? " · winner" : ""}${home}`);
    };
    $("vbars").onpointerleave = hideTip;
    $("vbars").onclick = (e) => { const j = barAt(e); if (j >= 0) { S.pinned = S.pinned === j ? -1 : j; dirty("all"); } };
    dirty("all");
  }

  // the probed neuron's voltage, its traces and the rule's drive at each step so far
  function voltTrace(j) {
    const r = VS.rec, T = S.T, n = S.nOut;
    const rate = (tau) => (tau > 0 ? 1 - Math.exp(-1 / tau) : 1);
    const rm = rate(r.tauMinus), rp = rate(r.tauPlus), kx = 1 - rate(r.tauX);
    const u = new Float32Array(T + 1), um = new Float32Array(T + 1), up = new Float32Array(T + 1), ltd = new Float32Array(T + 1), ltp = new Float32Array(T + 1);
    const inv = 1 / Math.max(1e-30, S.th[j] - r.rest), a = j < r.n ? r.aLtd[j] : r.aLtdBase;
    let m = 0, p = 0, x = 0;
    for (let t = 1; t <= S.t; t++) {
      const v = S.spk[t * n + j] ? r.vSpike : (S.trace[t * n + j] - r.rest) * inv;
      x = x * kx + S.nAct[t]; // Σ x̄ over the inputs, with this step's spikes
      // both read the traces as of the step before
      ltd[t] = a * Math.max(0, m - r.thMinus) * S.nAct[t];
      ltp[t] = r.aLtp * Math.max(0, v - r.thPlus) * Math.max(0, p - r.thMinus) * x;
      m += (v - m) * rm; p += (v - p) * rp;
      u[t] = v; um[t] = m; up[t] = p;
    }
    return { u, um, up, ltd, ltp };
  }

  function drawVolt() {
    const g = fit($("volt"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, r = VS.rec, L0 = 70, R0 = 12, T = S.T, n = S.nOut;
    x.font = `600 11px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    if (!r) { x.fillText("VOLTAGE-BASED STDP · WAITING FOR THE FIRST TRAINING IMAGE", 10, 16); VS.geom = null; return; }
    const j = probe(), tr = voltTrace(j);
    const c = (v) => +v.toPrecision(3);
    let sp = 0, sd = 0;
    for (let t = 1; t <= S.t; t++) { sp += tr.ltp[t]; sd += tr.ltd[t]; }
    const status = S.learn ? `SO FAR · LTP DRIVE ${fmtW(sp)} · LTD DRIVE ${fmtW(sd)}` : "TEST IMAGE · WEIGHTS FROZEN";
    // the constants go first when the panel is too narrow for both
    let title = `VOLTAGE-BASED STDP · ${nName(j)} · θ− ${c(r.thMinus)} · θ+ ${c(r.thPlus)} V_TH · τ− ${c(r.tauMinus)} · τ+ ${c(r.tauPlus)} · τX ${c(r.tauX)} MS`;
    if (x.measureText(title).width + x.measureText(status).width + 30 > w - R0) title = `VOLTAGE-BASED STDP · ${nName(j)}`;
    x.fillText(title, 10, 16);
    x.textAlign = "right"; x.fillStyle = G[2];
    if (x.measureText(title).width + x.measureText(status).width + 30 <= w - R0) x.fillText(status, w - R0, 16);
    const span = w - L0 - R0, cw = span / T, X = (t) => L0 + (t - 0.5) * cw;
    const vTop = 28, vH = Math.max(40, (h - vTop - 30) * 0.62), pTop = vTop + vH + 8, pH = h - pTop - 22;
    const lo = -0.4, hi = 1.5, Y = (v) => vTop + vH - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * vH;
    VS.geom = { L0, span, T, tr };
    x.textAlign = "left"; x.fillStyle = G[3];
    x.fillText("VOLTAGE", 10, vTop + vH / 2 + 4); x.fillText("LTP ↑", 10, pTop + pH / 4 + 4); x.fillText("LTD ↓", 10, pTop + (3 * pH) / 4 + 4);
    // inhibited steps, shaded as in the probe trace
    x.fillStyle = pa(0.05);
    for (let t = 1; t <= S.t; t++) if (!S.spk[t * n + j] && S.trace[t * n + j] <= S.floor + 1) x.fillRect(X(t) - cw / 2, vTop, cw + 0.5, vH);
    // where a pre spike would be depressed: ū− above θ−, as read at that step
    x.fillStyle = "rgba(236,131,90,0.22)";
    for (let t = 1; t <= S.t; t++) {
      const m = tr.um[t - 1];
      if (m > r.thMinus) { const y0 = Y(m), y1 = Y(r.thMinus); x.fillRect(X(t) - cw / 2, y0, cw + 0.5, y1 - y0); }
    }
    // rest, θ− and θ+
    x.fillStyle = pa(0.15); x.fillRect(L0, Math.round(Y(0)), span, 1);
    x.setLineDash([4, 4]); x.lineWidth = 1;
    x.strokeStyle = ORANGE; x.beginPath(); x.moveTo(L0, Y(r.thMinus)); x.lineTo(w - R0, Y(r.thMinus)); x.stroke();
    x.strokeStyle = pa(0.6); x.beginPath(); x.moveTo(L0, Y(r.thPlus)); x.lineTo(w - R0, Y(r.thPlus)); x.stroke();
    x.setLineDash([]);
    x.font = `600 10px ${FH}`; x.textAlign = "right";
    x.fillStyle = ORANGE; x.fillText("θ−", L0 - 4, Y(r.thMinus) + 4);
    x.fillStyle = G.p; x.fillText("θ+", L0 - 4, Y(r.thPlus) + 4);
    // u, with each spike drawn up to the top; then ū+ and ū−
    x.strokeStyle = G[3]; x.lineWidth = 1.2; x.beginPath();
    for (let t = 1; t <= S.t; t++) {
      const spike = S.spk[t * n + j], y = spike ? vTop : Y(tr.u[t]);
      t === 1 ? x.moveTo(X(t), y) : x.lineTo(X(t), y);
      if (spike) x.lineTo(X(t), Y(0));
    }
    x.stroke();
    for (const [arr, color] of [[tr.up, G.p], [tr.um, ORANGE]]) {
      x.strokeStyle = color; x.lineWidth = 1.6; x.beginPath();
      for (let t = 1; t <= S.t; t++) (t === 1 ? x.moveTo : x.lineTo).call(x, X(t), Y(arr[t]));
      x.stroke();
    }
    // plasticity: LTP up and LTD down, each on its own scale
    const mid = pTop + pH / 2;
    let pMax = 1e-12, dMax = 1e-12;
    for (let t = 1; t <= S.t; t++) { pMax = Math.max(pMax, tr.ltp[t]); dMax = Math.max(dMax, tr.ltd[t]); }
    x.fillStyle = pa(0.18); x.fillRect(L0, Math.round(mid), span, 1);
    for (let t = 1; t <= S.t; t++) {
      if (tr.ltp[t] > 0) { const bh = (tr.ltp[t] / pMax) * (pH / 2 - 2); x.fillStyle = G.p; x.fillRect(X(t) - Math.max(1, cw / 2), mid - bh, Math.max(2, cw), bh); }
      if (tr.ltd[t] > 0) { const bh = (tr.ltd[t] / dMax) * (pH / 2 - 2); x.fillStyle = ORANGE; x.fillRect(X(t) - cw / 2, mid + 1, Math.max(1, cw - 0.5), bh); }
    }
    // step axis and the legend
    x.font = `600 10px ${FH}`; x.textAlign = "center"; x.fillStyle = pa(0.5);
    for (let t = 50; t < T; t += 50) x.fillText(`${t}`, X(t), h - 6);
    x.textAlign = "right"; x.fillText("MS", w - R0, h - 6);
    let lx = 10;
    for (const [label, color] of [["u", G[3]], ["ū−", ORANGE], ["ū+", G.p]]) {
      x.fillStyle = color; x.fillRect(lx, h - 10, 12, 2); x.textAlign = "left"; x.fillText(label, lx + 15, h - 6); lx += 32;
    }
    if (VS.hover > 0) { x.fillStyle = pa(0.45); x.fillRect(Math.round(X(VS.hover)), vTop, 1, pTop + pH - vTop); }
  }

  // the probed neuron's weight change over the last training image
  function drawVdw() {
    const g = fit($("vdw"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, j = probe(), r = VS.rec, n = S.side;
    x.font = `600 11px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    x.fillText(`ΔW · ${nName(j)}`, 10, 16);
    const size = Math.min(w - 20, h - 62), ox = (w - size) / 2, oy = 24;
    x.strokeStyle = pa(0.2); x.lineWidth = 1; x.strokeRect(ox - 0.5, oy - 0.5, size + 1, size + 1);
    x.font = `12px ${FH}`;
    if (!r || j >= r.n) { x.fillStyle = pa(0.45); x.fillText("no training image yet", ox + 6, oy + 18); return; }
    const map = new Int8Array(S.nIn);
    for (let k = 0; k < r.inputs.length; k++) map[r.inputs[k]] = r.dw[k * r.n + j];
    drawGrid(x, ox, oy, size, n, n, (i) => { const b = 3 * (map[i] + 128); return [DIVERGE[b], DIVERGE[b + 1], DIVERGE[b + 2]]; });
    const moved = r.ltp[j] + r.ltd[j] > 0;
    x.fillStyle = G[2];
    x.fillText(VS.at === VS.images ? "this image" : "last training image", ox, oy + size + 15);
    x.fillStyle = pa(0.55);
    x.fillText(moved ? `±${fmtW(r.top[j])} · pale up, orange down` : "left unchanged", ox, oy + size + 30);
  }

  // what every neuron took from the last training image: the winner potentiates, the rivals
  // that came close to firing are depressed
  function drawVbars() {
    const g = fit($("vbars"));
    if (!g || !S.meta) return;
    const { x, w, h } = g, r = VS.rec;
    x.font = `600 11px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    x.fillText(`LAST TRAINING IMAGE${VS.label >= 0 ? ` · LABEL ${VS.label}` : ""}`, 10, 16);
    if (!r) { VS.bars = null; return; }
    const N = r.n, L = 10, bw = (w - 20) / N, top = 28, bottom = h - 36, mid = top + (bottom - top) * 0.6;
    let pMax = 1e-12, dMax = 1e-12;
    for (let j = 0; j < N; j++) { pMax = Math.max(pMax, r.ltp[j]); dMax = Math.max(dMax, r.ltd[j]); }
    VS.bars = { L, bw, N };
    x.fillStyle = pa(0.18); x.fillRect(L, Math.round(mid), w - 20, 1);
    const pj = probe();
    for (let j = 0; j < N; j++) {
      const bx = L + j * bw, ww = Math.max(1, bw - (bw > 4 ? 1 : 0));
      const ph = (r.ltp[j] / pMax) * (mid - top - 2), dh = (r.ltd[j] / dMax) * (bottom - mid - 2);
      x.fillStyle = j === VS.winner ? G.p : gb(0.8); x.fillRect(bx, mid - ph, ww, ph);
      x.fillStyle = ORANGE; x.fillRect(bx, mid + 1, ww, dh);
      if (j === pj) { x.strokeStyle = G.p; x.lineWidth = 1; x.strokeRect(bx - 1.5, top - 1.5, ww + 3, bottom - top + 3); }
    }
    x.font = `11px ${FH}`; x.fillStyle = G[2];
    x.fillText(`LTP ↑ TO ${fmtW(pMax)} · LTD ↓ TO ${fmtW(dMax)}`, 10, h - 20);
    let note = "homeostasis off · LTD fixed";
    if (r.uRef > 0) {
      let a = Infinity, b = 0;
      for (let j = 0; j < N; j++) { const k = r.aLtd[j] / r.aLtdBase; a = Math.min(a, k); b = Math.max(b, k); }
      note = `homeostasis · LTD ×${a.toFixed(2)}–${b.toFixed(2)}`;
    }
    x.fillStyle = pa(0.55); x.fillText(note, 10, h - 6);
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
      DA.on = false; DA.rec = null; DA.tags.clear();
      VS.on = false; VS.rec = null;
      document.querySelector(".app").style.gridTemplateRows = "";
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
      S.nAct = new Uint16Array(T + 1);
      DA.rec = null; DA.tags.clear(); VS.rec = null;
      dirty("all");
    },
    image() {
      S.trace.fill(0); S.spk.fill(0); S.peak.fill(0); S.nAct.fill(0); VS.images++; S.t = 0; S.winner = -1; S.result = null; S.inSpikes = 0; S.outSpikes = 0;
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
        S.inSpikes += na; S.outSpikes += nf; S.nAct[t] = na; S.t = t;
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
    14(v, p) { // DOPAMINE
      const r = { clock: v.getFloat64(p, true), steps: v.getUint16(p + 8, true), gap: v.getUint32(p + 10, true), delay: v.getUint32(p + 14, true),
        jitter: v.getUint32(p + 18, true), tauC: v.getFloat32(p + 22, true), tauD: v.getFloat32(p + 26, true), gain: v.getFloat32(p + 30, true),
        level: v.getFloat32(p + 34, true), live: v.getUint16(p + 38, true), trials: [] };
      const n = v.getUint8(p + 40);
      for (let k = 0, o = p + 41; k < n; k++, o += 34) {
        r.trials.push({ start: v.getFloat64(o, true), label: v.getUint8(o + 8), winner: v.getInt16(o + 9, true), correct: v.getUint8(o + 11) === 1,
          decided: v.getUint16(o + 12, true), amount: v.getFloat32(o + 14, true), at: v.getFloat64(o + 18, true), tagged: v.getFloat32(o + 26, true),
          credit: v.getFloat32(o + 30, true) });
      }
      // a puff of r is scaled so a tag made as it lands meets r of it, as dopamine.rs does
      r.kd = Math.exp(-1 / Math.max(r.tauD, 1e-9));
      r.pulse = r.gain * (1 - Math.exp(-1 / Math.max(r.tauC, 1e-9)) * r.kd);
      const fresh = !DA.rec || r.clock !== DA.rec.clock;
      DA.rec = r;
      if (fresh) DA.shown = r.clock - r.gap;
      if (!DA.on) enableDopamine();
      dirty("dopa");
    },
    15(v, p) { // TAGS
      const rows = v.getUint16(p, true), n = v.getUint16(p + 2, true);
      DA.tags.clear();
      for (let k = 0, o = p + 4; k < rows; k++, o += 14 + n) {
        DA.tags.set(v.getUint16(o, true), { scale: v.getFloat32(o + 2, true), size: v.getFloat32(o + 6, true), age: v.getFloat32(o + 10, true),
          data: new Int8Array(v.buffer.slice(v.byteOffset + o + 14, v.byteOffset + o + 14 + n)) });
      }
      dirty("tags", "weights");
    },
    51(v, p) { // VOLTAGE
      const f = (k) => v.getFloat32(p + 4 * k, true);
      const r = { tauMinus: f(0), tauPlus: f(1), tauX: f(2), thMinus: f(3), thPlus: f(4), vSpike: f(5), aLtp: f(6), aLtdBase: f(7), uRef: f(8), rest: f(9),
        n: v.getUint16(p + 40, true) };
      const m = v.getUint16(p + 42, true), N = r.n;
      for (const k of ["aLtd", "bar", "ltp", "ltd", "top"]) r[k] = new Float32Array(N);
      let o = p + 44;
      for (let j = 0; j < N; j++, o += 20) {
        r.aLtd[j] = v.getFloat32(o, true); r.bar[j] = v.getFloat32(o + 4, true); r.ltp[j] = v.getFloat32(o + 8, true);
        r.ltd[j] = v.getFloat32(o + 12, true); r.top[j] = v.getFloat32(o + 16, true);
      }
      r.inputs = new Uint16Array(m);
      for (let k = 0; k < m; k++, o += 2) r.inputs[k] = v.getUint16(o, true);
      r.dw = new Int8Array(v.buffer.slice(v.byteOffset + o, v.byteOffset + o + m * N));
      // it follows the RESULT of the image it learned from
      VS.rec = r; VS.at = VS.images; VS.winner = S.result ? S.result.winner : -1; VS.label = S.result ? S.label : -1;
      if (!VS.on) enableVoltage();
      dirty("volt", "vdw", "vbars");
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
      if (DA.on) {
        const moving = daTick();
        if (all || moving || S.dirty.has("dopa") || S.dirty.has("raster")) drawDopa();
        if (all || moving || S.dirty.has("tags") || S.dirty.has("trace")) drawTags();
      }
      if (VS.on) {
        if (all || S.dirty.has("volt") || S.dirty.has("trace")) drawVolt();
        if (all || S.dirty.has("vdw") || S.dirty.has("trace")) drawVdw();
        if (all || S.dirty.has("vbars") || S.dirty.has("trace")) drawVbars();
      }
    },
  });
}
