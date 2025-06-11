// PC-SNN view (mnist-pcsnn-rs). Fig. 01 replays one image's first spikes:
// input pixels in firing order, hidden neurons as they cross threshold, output
// spike times on a timeline. After the replay, record 48 shows what inference
// did: hidden times moved earlier (green) or later (orange), the output targets,
// and the free energy over the iterations.
"use strict";
{
  const M = { hidT: null, outT: null, inf: null, boxes: null, rows: null };
  const MS_PER_ITER = 40;

  // earlier spikes are paler
  const early = (t, T) => { const f = 1 - Math.min(1, t / T); return [24 + 214 * f, 74 + 168 * f, 51 + 187 * f]; };

  function arrow(x, x0, y0, x1, y1, label, dashed) {
    x.strokeStyle = dashed ? "rgba(236,131,90,0.85)" : pa(0.7); x.fillStyle = x.strokeStyle; x.lineWidth = 1.2;
    if (dashed) x.setLineDash([5, 4]);
    x.beginPath(); x.moveTo(x0, y0); x.lineTo(x1, y1); x.stroke(); x.setLineDash([]);
    const a = Math.atan2(y1 - y0, x1 - x0);
    x.beginPath(); x.moveTo(x1, y1); x.lineTo(x1 - 7 * Math.cos(a - 0.4), y1 - 7 * Math.sin(a - 0.4)); x.lineTo(x1 - 7 * Math.cos(a + 0.4), y1 - 7 * Math.sin(a + 0.4)); x.fill();
    x.font = `600 11px ${FH}`; x.textAlign = "center"; x.fillText(label, (x0 + x1) / 2, Math.min(y0, y1) - 6);
  }

  function draw(x, w, h, P) {
    rulers(x, w, h);
    if (!M.hidT) return;
    const T = S.T, now = Math.max(0, P.t - 1), n = S.side, hq = S.pops[1], C = S.meta.classes;
    const pad = 28, top = 40, s = Math.max(60, Math.min(h - top - 40, (w - 2 * pad) * 0.27)), gap = s * 0.3;
    const y0 = top + (h - top - s) / 2;
    const xi = pad, xh = xi + s + gap, xo = xh + s + gap, wo = w - pad - xo;

    // input: pixels light up when their spike time has come
    const tin = (px) => Math.round(((255 - px) / 255) * T);
    drawGrid(x, xi, y0, s, n, n, (i) => {
      const t = tin(S.img[i]);
      if (t >= T) return [24, 66, 46];
      if (t > now) return [30, 80, 58];
      return early(t, 60);
    });
    // hidden: filled once fired; after inference a ring shows where it was moved
    const inf = M.inf;
    // the iterations play out over the time the trainer holds for them
    const it = inf ? Math.min(inf.energy.length - 1, Math.floor((performance.now() - inf.at) / MS_PER_ITER)) : 0;
    if (inf && it < inf.energy.length - 1) dirty("net");
    const hidAt = (j) => (inf ? inf.trail[it * inf.nh + j] : 0);
    const hs = s, cs = drawGrid(x, xh, y0, hs, hq.w, hq.h, (j) => {
      if (j >= hq.size) return null;
      if (inf && inf.drop[j]) return [20, 44, 32];
      const t = M.hidT[j];
      return t >= 0 && t <= now ? early(t, 120) : [24, 66, 46];
    });
    if (inf) {
      for (let j = 0; j < hq.size; j++) {
        if (inf.drop[j]) continue;
        const d = hidAt(j) - inf.pred[j];
        if (Math.abs(d) < 0.5) continue;
        const cx = xh + ((j % hq.w) + 0.5) * cs, cy = y0 + (((j / hq.w) | 0) + 0.5) * cs;
        x.strokeStyle = d < 0 ? gb(0.95) : ORANGE; x.lineWidth = Math.min(3, 0.6 + Math.abs(d) / 10);
        x.strokeRect(cx - cs / 2 + 1, cy - cs / 2 + 1, cs - 2, cs - 2);
      }
    }
    M.boxes = [{ pop: 1, ox: xh, oy: y0, cs, q: hq }];
    if (P.probe.pop === 1) {
      const u = P.probe.unit, cx = xh + ((u % hq.w) + 0.5) * cs, cy = y0 + (((u / hq.w) | 0) + 0.5) * cs;
      x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, cx, cy, cs * 0.9, 3);
    }

    // outputs on a timeline, zoomed to where the spikes are
    let tEnd = 40;
    for (let k = 0; k < C; k++) {
      for (const t of [M.outT[k], inf && inf.out[k][0], inf && inf.out[k][2]]) if (t >= 0 && t < T) tEnd = Math.max(tEnd, t + 25);
    }
    tEnd = Math.min(T, Math.max(tEnd, now + 5));
    const rh = s / C, X = (t) => xo + 22 + (Math.min(t, tEnd) / tEnd) * (wo - 30);
    M.rows = { xo, y0, rh, X };
    const o = P.outcome, winner = o ? o.predicted : -1, label = o ? o.label : S.label;
    if (inf) { // gamma band: where the wrong classes are pushed to
      let first = T;
      for (let k = 0; k < C; k++) first = Math.min(first, inf.out[k][0]);
      const g = S.params.find((q) => q.name === "γ margin");
      if (first < T && g) { x.fillStyle = "rgba(236,131,90,0.07)"; x.fillRect(X(first), y0, X(first + g.value) - X(first), s); }
    }
    x.font = `600 12px ${FH}`;
    for (let k = 0; k < C; k++) {
      const cy = y0 + (k + 0.5) * rh;
      x.fillStyle = pa(0.08); x.fillRect(xo + 22, Math.round(cy) + 0.5, wo - 30, 1);
      x.textAlign = "right"; x.fillStyle = k === label ? G.p : G[2]; x.fillText(k, xo + 14, cy + 4);
      const t = M.outT[k];
      if (t >= 0 && t <= now) { x.beginPath(); x.arc(X(t), cy, 4, 0, 7); x.fillStyle = k === winner ? G.p : G[4]; x.fill(); }
      if (inf) {
        const [, pred, target] = inf.out[k];
        if (target < T) { // target time inference held this output to
          x.strokeStyle = k === label ? G.p : ORANGE; x.lineWidth = 1.2; const r = 5, tx = X(target);
          x.beginPath(); x.moveTo(tx, cy - r); x.lineTo(tx + r, cy); x.lineTo(tx, cy + r); x.lineTo(tx - r, cy); x.closePath(); x.stroke();
        }
        if (pred < T) { x.fillStyle = ORANGE; x.fillRect(X(pred) - 0.5, cy - 6, 1.5, 12); }
      }
      if (k === winner) { x.strokeStyle = G.p; x.lineWidth = 1; bracket(x, X(M.outT[k] >= 0 ? M.outT[k] : 0), cy, 8, 3); }
      if (P.probe.pop === 2 && P.probe.unit === k) { x.strokeStyle = G.p; x.lineWidth = 1.5; x.strokeRect(xo + 2, cy - rh / 2 + 1, 16, rh - 2); }
    }
    x.fillStyle = pa(0.5); x.fillRect(Math.round(X(now)) + 0.5, y0 - 4, 1, s + 8);
    x.font = `11px ${FH}`; x.fillStyle = G[2]; x.textAlign = "left"; x.fillText("0", X(0) - 3, y0 + s + 14);
    x.textAlign = "right"; x.fillText(`${Math.round(tEnd)} steps`, X(tEnd), y0 + s + 14);

    x.font = `600 12px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    x.fillText("INPUT · FIRST SPIKE BY BRIGHTNESS", xi, y0 - 12);
    x.fillText(`HIDDEN · ${hq.size} IF`, xh, y0 - 12);
    x.fillText("OUTPUT SPIKE TIMES · ● fired ◇ target ▏after inference", xo + 22, y0 - 12);
    arrow(x, xi + s + 4, y0 + s * 0.35, xh - 6, y0 + s * 0.35, "t̂ = IF(W¹·spikes)", false);
    arrow(x, xh + s + 4, y0 + s * 0.35, xo + 4, y0 + s * 0.35, "W²", false);
    arrow(x, xo + 4, y0 + s * 0.68, xh + s + 4, y0 + s * 0.68, "ε · inference", true);

    // free energy over the inference iterations
    if (inf && inf.energy.length > 1) {
      const fx = xh, fw = s, fy = y0 + s + 26, fh = Math.max(16, Math.min(40, h - fy - 6)), e = inf.energy;
      const hi = Math.max(...e, 1e-6);
      x.strokeStyle = ORANGE; x.lineWidth = 1.5; x.beginPath();
      e.forEach((v, i) => { const px = fx + (i / (e.length - 1)) * fw, py = fy + fh - (v / hi) * fh; if (i) x.lineTo(px, py); else x.moveTo(px, py); });
      x.stroke();
      const cx = fx + (it / (e.length - 1)) * fw;
      x.beginPath(); x.arc(cx, fy + fh - (e[it] / hi) * fh, 3, 0, 7); x.fillStyle = G.p; x.fill();
      x.font = `600 10px ${FH}`; x.fillStyle = ORANGE; x.textAlign = "left";
      x.fillText(`FREE ENERGY ${e[0].toFixed(1)} → ${e[e.length - 1].toFixed(1)} OVER ${e.length - 1} ITERATIONS`, fx, fy - 4);
    }
  }

  function rowHit(mx, my) {
    const r = M.rows;
    if (!r || mx < r.xo) return -1;
    const k = Math.floor((my - r.y0) / r.rh);
    return k >= 0 && k < S.meta.classes ? k : -1;
  }

  registerView("pcsnn", popView({
    fig1: "Fig. 01 — spike-time inference · click a neuron to probe",
    draw,
    hit(mx, my) {
      const u = hitPops(M.boxes, mx, my);
      if (u) return u;
      const k = rowHit(mx, my);
      return k >= 0 ? [2, k] : null;
    },
    tip(mx, my) {
      const u = hitPops(M.boxes, mx, my), inf = M.inf;
      if (u) {
        const j = u[1], t = M.hidT[j];
        const fired = t >= 0 ? `fired at ${t}` : "silent";
        const moved = inf && !inf.drop[j] ? ` · inferred ${inf.hid[j].toFixed(1)}` : inf && inf.drop[j] ? " · dropped" : "";
        return [`hidden N${j}`, ` · ${fired}${moved}`];
      }
      const k = rowHit(mx, my);
      if (k < 0) return null;
      const t = M.outT[k], tail = inf ? ` · target ${inf.out[k][2].toFixed(0)} · after inference ${inf.out[k][1].toFixed(0)}` : "";
      return [`output ${k}`, ` · ${t >= 0 ? `fired at ${t}` : "silent"}${tail}`];
    },
    pops() { M.hidT = new Int16Array(S.pops[1].size); M.outT = new Int16Array(S.meta.classes); M.inf = null; },
    image() { M.hidT.fill(-1); M.outT.fill(-1); M.inf = null; },
    step(t, lists) {
      for (const j of lists[1]) M.hidT[j] = t - 1;
      for (const k of lists[2]) M.outT[k] = t - 1;
    },
    H: {
      48(v, p) { // INFER
        const n = v.getUint8(p), energy = [];
        let o = p + 1;
        for (let i = 0; i < n; i++, o += 4) energy.push(v.getFloat32(o, true));
        const nh = v.getUint16(o, true); o += 2;
        const pred = new Float32Array(nh), hid = new Float32Array(nh), drop = new Uint8Array(nh);
        for (let j = 0; j < nh; j++, o += 9) { pred[j] = v.getFloat32(o, true); hid[j] = v.getFloat32(o + 4, true); drop[j] = v.getUint8(o + 8); }
        const trail = new Float32Array(v.buffer.slice(v.byteOffset + o, v.byteOffset + o + n * nh * 4)); o += n * nh * 4;
        const nc = v.getUint8(o++), out = [];
        for (let k = 0; k < nc; k++, o += 12) out.push([v.getFloat32(o, true), v.getFloat32(o + 4, true), v.getFloat32(o + 8, true)]);
        M.inf = { energy, pred, hid, drop, out, trail, nh, at: performance.now() };
        dirty("net");
      },
    },
  }));
}
