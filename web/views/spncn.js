// SpNCN view (mnist-spncn-rs). Fig. 01: the input trace x, the network's
// prediction ẑ of it and their difference e — the error units — then the latent
// layers, with W carrying predictions down and E carrying errors up.
"use strict";
{
  const M = { trace: null, pred: null, settled: null, decay: 0.967, peak: [], boxes: [], maps: null, last: 0 };
  const PRED = 0, SETTLED = 1;

  // signed value on a ±scale ramp, as an rgb triple
  const ramp = (v, s) => { const q = Math.max(1, Math.min(255, Math.round(128 + (v / s) * 127))) * 3; return [DIVERGE[q], DIVERGE[q + 1], DIVERGE[q + 2]]; };
  const unpack = (m) => { const out = new Float32Array(m.w * m.h); for (let i = 0; i < out.length; i++) out[i] = m.signed ? ((m.data[i] - 128) / 127) * m.scale : (m.data[i] / 255) * m.scale; return out; };

  function arrowHead(x, x1, y1, ang) {
    x.beginPath(); x.moveTo(x1, y1);
    x.lineTo(x1 - 7 * Math.cos(ang - 0.4), y1 - 7 * Math.sin(ang - 0.4));
    x.lineTo(x1 - 7 * Math.cos(ang + 0.4), y1 - 7 * Math.sin(ang + 0.4));
    x.closePath(); x.fill();
  }

  // dashed straight arrow: errors going up
  function errArrow(x, x0, x1, y, label) {
    x.strokeStyle = gb(0.8); x.fillStyle = gb(0.8); x.lineWidth = 1.2; x.setLineDash([5, 4]);
    x.beginPath(); x.moveTo(x0, y); x.lineTo(x1 - 2, y); x.stroke(); x.setLineDash([]);
    arrowHead(x, x1, y, 0);
    x.font = `600 11px ${FH}`; x.textAlign = "center"; x.fillText(label, (x0 + x1) / 2, y + 16);
  }

  // solid arc over the top: predictions going down
  function predArc(x, x0, x1, y, label) {
    const mid = (x0 + x1) / 2, lift = Math.min(46, (x0 - x1) * 0.35);
    x.strokeStyle = G.p; x.fillStyle = G.p; x.lineWidth = 1.2;
    x.beginPath(); x.moveTo(x0, y); x.quadraticCurveTo(mid, y - lift * 2, x1, y); x.stroke();
    arrowHead(x, x1, y, Math.atan2(lift * 2, x1 - mid));
    x.font = `600 11px ${FH}`; x.textAlign = "center"; x.fillText(label, mid, y - lift - 4);
  }

  function diamond(x, cx, cy, r, glow) {
    x.beginPath(); x.moveTo(cx, cy - r); x.lineTo(cx + r, cy); x.lineTo(cx, cy + r); x.lineTo(cx - r, cy); x.closePath();
    x.fillStyle = `rgba(82,196,149,${0.25 + 0.75 * glow})`; x.fill();
    x.strokeStyle = G.p; x.lineWidth = 1; x.stroke();
  }

  function draw(x, w, h, P) {
    rulers(x, w, h);
    if (!M.trace) return;
    const ps = S.pops, nl = ps.length - 1, n = S.side, nS = P.series.step.length;
    const pad = 28, top = 34;
    const units = 3 + nl * 1.15 + 0.35 * (2 + 2 * nl);
    const s = Math.max(40, Math.min(h - top - 70, (w - 2 * pad) / units)), gap = s * 0.35;
    const y0 = top + (h - top - s) / 2 + 6;
    const t = Math.max(1, Math.min(P.t, S.T));
    const pred = M.settled && P.outcome ? M.settled : M.pred;
    const err = new Float32Array(M.trace.length);
    let tmax = 1e-6, emax = 1e-6;
    for (let i = 0; i < err.length; i++) {
      err[i] = M.trace[i] - (pred ? pred[i] : 0);
      tmax = Math.max(tmax, M.trace[i]); emax = Math.max(emax, Math.abs(err[i]));
    }
    const scale = Math.max(tmax, pred ? Math.max(...pred.map(Math.abs)) : 0, 1);
    const xs = [pad, pad + s + gap, pad + 2 * (s + gap)];
    M.maps = { xs, y0, s, n, err };
    const glowIn = P.glow[0];
    drawGrid(x, xs[0], y0, s, n, n, (i) => { const c = ramp(M.trace[i], scale); const g = glowIn[i]; return g > 0.05 ? [c[0] + (238 - c[0]) * g, c[1] + (242 - c[1]) * g, c[2] + (238 - c[2]) * g] : c; });
    drawGrid(x, xs[1], y0, s, n, n, (i) => ramp(pred ? pred[i] : 0, scale));
    drawGrid(x, xs[2], y0, s, n, n, (i) => ramp(err[i], scale));
    x.font = `600 12px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
    x.fillText("INPUT TRACE x", xs[0], y0 - 10);
    x.fillText(pred === M.settled ? "PREDICTION ẑ · SETTLED" : "PREDICTION ẑ = W¹·z¹", xs[1], y0 - 10);
    x.fillText("ERROR UNITS e⁰ = x − ẑ", xs[2], y0 - 10);
    x.font = `600 22px ${FH}`; x.textAlign = "center"; x.fillStyle = pa(0.6);
    x.fillText("−", xs[0] + s + gap / 2, y0 + s / 2 + 7); x.fillText("=", xs[1] + s + gap / 2, y0 + s / 2 + 7);

    // latent layers, each with the error units below it as a diamond
    let px = xs[2] + s;
    M.boxes = [];
    for (let l = 1; l <= nl; l++) {
      const q = ps[l], gs = l === 1 ? s : s * 0.72, gx = px + gap * (l === 1 ? 1.4 : 2.2), gy = y0 + (s - gs) / 2;
      if (l > 1) {
        const en = P.strace[t * nS + (l - 1)] || 0;
        M.peak[l - 1] = Math.max(M.peak[l - 1] || 1e-9, en);
        const dx = px + gap * 1.1;
        diamond(x, dx, y0 + s / 2, 7 + 9 * Math.sqrt(en / M.peak[l - 1]), en / M.peak[l - 1]);
        x.font = `600 11px ${FH}`; x.textAlign = "center"; x.fillStyle = G[3]; x.fillText(`e${"⁰¹²³⁴"[l - 1]}`, dx, y0 + s / 2 + 28);
        errArrow(x, dx + 18, gx - 4, y0 + s / 2, `E${"¹²³⁴⁵"[l - 1]}`);
        errArrow(x, px + 4, dx - 18, y0 + s / 2, "");
        predArc(x, gx + gs * 0.3, dx, gy - 4, `W${"¹²³⁴⁵"[l - 1]}`);
      } else {
        errArrow(x, px + 6, gx - 4, y0 + s / 2, "E¹ · errors up");
        predArc(x, gx + gs * 0.3, xs[1] + s * 0.6, Math.min(gy, y0) - 4, "W¹ · prediction down");
      }
      const glow = P.glow[l];
      const cs = drawGrid(x, gx, gy, gs, q.w, q.h, (u) => {
        if (u >= q.size) return null;
        const g = glow[u];
        return g > 0.02 ? [18 + 220 * g, 53 + 189 * g, 36 + 202 * g] : [24, 66, 46];
      });
      x.font = `600 12px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
      x.fillText(`${q.name} · ${q.size}`, gx, gy - 10);
      if (l === P.probe.pop) {
        const u = P.probe.unit, cx = gx + ((u % q.w) + 0.5) * cs, cy = gy + (((u / q.w) | 0) + 0.5) * cs;
        x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, cx, cy, Math.max(4, cs * 0.9), 3);
      }
      M.boxes.push({ pop: l, ox: gx, oy: gy, cs, q });
      px = gx + gs;
    }
    // input error energy over this presentation, under the maps
    const e0 = P.strace, by = y0 + s + 18, bw = xs[2] + s - xs[0], bh = Math.max(14, Math.min(34, h - by - 8));
    let em = 1e-9;
    for (let k = 1; k <= S.T; k++) em = Math.max(em, e0[k * nS]);
    x.fillStyle = pa(0.08); x.fillRect(xs[0], by + bh, bw, 1);
    x.strokeStyle = ORANGE; x.lineWidth = 1.2; x.beginPath();
    for (let k = 1; k <= t; k++) { const X = xs[0] + ((k - 1) / (S.T - 1)) * bw, Y = by + bh - (e0[k * nS] / em) * bh; if (k === 1) x.moveTo(X, Y); else x.lineTo(X, Y); }
    x.stroke();
    x.font = `600 10px ${FH}`; x.textAlign = "right"; x.fillStyle = ORANGE; x.fillText("‖e⁰‖² SETTLING OVER THE IMAGE", xs[0] + bw, by - 2);
  }

  const mapHit = (mx, my) => {
    const g = M.maps;
    if (!g) return -1;
    for (let k = 0; k < 3; k++) {
      const c = Math.floor((mx - g.xs[k]) / (g.s / g.n)), r = Math.floor((my - g.y0) / (g.s / g.n));
      if (c >= 0 && c < g.n && r >= 0 && r < g.n) return r * g.n + c;
    }
    return -1;
  };

  registerView("spncn", popView({
    fig1: "Fig. 01 — predictive hierarchy · click a unit to probe",
    draw,
    hit: (mx, my) => hitPops(M.boxes, mx, my),
    tip(mx, my) {
      const u = hitPops(M.boxes, mx, my);
      if (u) return [`${S.pops[u[0]].name} · N${u[1]}`, " · click to probe"];
      const i = mapHit(mx, my);
      if (i < 0 || !M.trace) return null;
      const p = M.pred ? M.pred[i] : 0;
      return [`e ${M.maps.err[i].toFixed(2)}`, ` · x ${M.trace[i].toFixed(2)} · ẑ ${p.toFixed(2)} · pixel ${i % S.side},${(i / S.side) | 0}`];
    },
    pops() {
      M.trace = new Float32Array(S.nIn); M.pred = null; M.settled = null; M.peak = [];
      const tau = S.params.find((q) => q.name === "τ trace");
      M.decay = 1 - (S.meta.dt || 1) / (tau ? tau.value : 30);
    },
    image() { M.trace.fill(0); M.settled = null; },
    step(t, lists) {
      for (let i = 0; i < M.trace.length; i++) M.trace[i] *= M.decay;
      for (const i of lists[0]) M.trace[i] += 1;
    },
    map(id, m) {
      if (id === PRED) M.pred = unpack(m);
      else if (id === SETTLED) M.settled = unpack(m);
    },
  }));
}
