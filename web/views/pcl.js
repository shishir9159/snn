// Predictive Coding Light view (mnist-pcl-rs). Fig. 01: the event retina as
// the eye makes three saccades (ON green, OFF red), simple cells as
// hypercolumns of features per location, complex cells above them, and the
// paper's pathway colours — green excitation, orange plastic lateral
// inhibition, magenta static inhibition, blue plastic top-down inhibition.
"use strict";
{
  const BLUE = "#7fb2e5", MAGENTA = "#d07fc8";
  const M = { boxes: [], lay: null };

  // simple pop is laid out as n*K/4 by 4n, complex as 4m by m*Kc/4
  function shape() {
    const s = S.pops[2], c = S.pops[3];
    const n = s.h / 4, k = (s.w * 4) / n, m = c.w / 4, kc = (c.h * 4) / m;
    return { n, k, m, kc };
  }

  const param = (name, d) => { const q = S.params.find((x) => x.name === name); return q ? q.value : d; };

  // eye position at step t along the triangle: right, down-left, home
  function eye(t) {
    const a = param("Saccade px", 3), leg = param("Saccade leg", 100);
    const c = [[0, 0], [a, 0], [a / 2, a], [0, 0]], l = Math.min(2, Math.floor(t / leg)), f = Math.min(1, (t - l * leg) / leg);
    return [c[l][0] + (c[l + 1][0] - c[l][0]) * f, c[l][1] + (c[l + 1][1] - c[l][1]) * f];
  }

  function label(x, text, px, py, color) { x.font = `600 12px ${FH}`; x.textAlign = "left"; x.fillStyle = color || G[3]; x.fillText(text, px, py); }

  function draw(x, w, h, P) {
    rulers(x, w, h);
    if (!S.pops[3]) return;
    const { n, k, m, kc } = shape(), side = S.pops[0].w, t = Math.max(0, P.t);
    const pad = 28, top = 40, s = Math.max(60, Math.min(h - top - 46, (w - 2 * pad) * 0.3)), gap = s * 0.28;
    const y0 = top + (h - top - s) / 2 - 8, xr = pad, xs = xr + s + gap, xc = xs + s + gap;
    const sc = Math.min(s, w - pad - xc);

    // retina
    const on = P.glow[0], off = P.glow[1];
    drawGrid(x, xr, y0, s, side, side, (i) => [18 + 218 * off[i], 53 + 189 * on[i], 36 + 40 * on[i]]);
    label(x, `RETINA ${side}×${side} · ON / OFF EVENTS`, xr, y0 - 10);
    // saccade path with the eye now
    const a = param("Saccade px", 3), q = 14, ix = xr + s - 3 * q - 6, iy = y0 + 6, sq = (3 * q) / a;
    x.strokeStyle = pa(0.35); x.lineWidth = 1; x.beginPath(); x.moveTo(ix, iy); x.lineTo(ix + a * sq, iy); x.lineTo(ix + (a / 2) * sq, iy + a * sq); x.closePath(); x.stroke();
    const [ex, ey] = eye(t); x.beginPath(); x.arc(ix + ex * sq, iy + ey * sq, 3, 0, 7); x.fillStyle = G.p; x.fill();

    // simple hypercolumns: K features per location in 8 x K/8 cells
    const fw = 8, fh = k / fw, hb = s / n, cell = (hb - 4) / Math.max(fw, fh);
    const glowS = P.glow[2];
    M.boxes = [];
    for (let ly = 0; ly < n; ly++) for (let lx = 0; lx < n; lx++) {
      const bx = xs + lx * hb + 2, by = y0 + ly * hb + 2 + (hb - 4 - cell * fh) / 2;
      x.strokeStyle = "rgba(208,127,200,0.35)"; x.lineWidth = 1; x.strokeRect(bx - 1, by - 1, cell * fw + 2, cell * fh + 2);
      for (let f = 0; f < k; f++) {
        const u = (ly * n + lx) * k + f, g = glowS[u];
        x.fillStyle = g > 0.02 ? `rgb(${18 + 220 * g},${53 + 189 * g},${36 + 202 * g})` : "rgb(24,66,46)";
        x.fillRect(bx + (f % fw) * cell, by + Math.floor(f / fw) * cell, cell - 1, cell - 1);
      }
      M.boxes.push({ pop: 2, x: bx, y: by, cell, fw, base: (ly * n + lx) * k, count: k });
    }
    label(x, `SIMPLE · ${n}² LOCATIONS × ${k} FEATURES`, xs, y0 - 10);

    // complex: kc cells per location in 4 x kc/4
    const cb = sc / m, cw = 4, ch = kc / cw, ccell = (cb - 6) / Math.max(cw, ch), yc = y0 + (s - sc) / 2;
    const glowC = P.glow[3];
    for (let ly = 0; ly < m; ly++) for (let lx = 0; lx < m; lx++) {
      const bx = xc + lx * cb + 3, by = yc + ly * cb + 3;
      for (let c = 0; c < kc; c++) {
        const u = (ly * m + lx) * kc + c, g = glowC[u];
        x.fillStyle = g > 0.02 ? `rgb(${18 + 220 * g},${53 + 189 * g},${36 + 202 * g})` : "rgb(24,66,46)";
        x.fillRect(bx + (c % cw) * ccell, by + Math.floor(c / cw) * ccell, ccell - 1, ccell - 1);
      }
      M.boxes.push({ pop: 3, x: bx, y: by, cell: ccell, fw: cw, base: (ly * m + lx) * kc, count: kc });
    }
    label(x, `COMPLEX · ${m}² × ${kc}`, xc, yc - 10);

    // pathways
    const arrow = (x0, y1, x1, color, text, dy) => {
      x.strokeStyle = color; x.fillStyle = color; x.lineWidth = 1.4; x.beginPath(); x.moveTo(x0, y1); x.lineTo(x1, y1); x.stroke();
      const d = Math.sign(x1 - x0); x.beginPath(); x.moveTo(x1, y1); x.lineTo(x1 - 7 * d, y1 - 4); x.lineTo(x1 - 7 * d, y1 + 4); x.fill();
      x.font = `600 10px ${FH}`; x.textAlign = "center"; x.fillText(text, (x0 + x1) / 2, y1 + dy);
    };
    arrow(xr + s + 4, y0 + s * 0.3, xs - 4, G[4], "eSTDP", -6);
    arrow(xs + s + 4, y0 + s * 0.3, xc - 4, G[4], "eSTDP", -6);
    arrow(xc - 4, y0 + s * 0.62, xs + s + 4, BLUE, "top-down iSTDP", 14);
    x.strokeStyle = "rgba(236,131,90,0.8)"; x.lineWidth = 1.2; x.beginPath();
    x.moveTo(xs + hb * 0.5, y0 + s + 6); x.quadraticCurveTo(xs + hb, y0 + s + 22, xs + hb * 1.5, y0 + s + 6); x.stroke();
    x.font = `600 10px ${FH}`; x.textAlign = "left"; x.fillStyle = ORANGE; x.fillText("lateral iSTDP", xs + hb * 1.7, y0 + s + 18);
    x.fillStyle = MAGENTA; x.fillText("static", xs + hb * 3.4, y0 + s + 18);

    // probe: its cell and, for a simple cell, its receptive field on the retina
    const pr = P.probe;
    for (const b of M.boxes) if (b.pop === pr.pop && pr.unit >= b.base && pr.unit < b.base + b.count) {
      const f = pr.unit - b.base, cx = b.x + (f % b.fw + 0.5) * b.cell, cy = b.y + (Math.floor(f / b.fw) + 0.5) * b.cell;
      x.strokeStyle = G.p; x.lineWidth = 1.5; bracket(x, cx, cy, b.cell, 3);
    }
    if (pr.pop === 2) {
      const loc = Math.floor(pr.unit / k), lx = loc % n, ly = Math.floor(loc / n), cs = s / side;
      x.strokeStyle = G.p; x.setLineDash([3, 3]); x.strokeRect(xr + lx * 6 * cs, y0 + ly * 6 * cs, 10 * cs, 10 * cs); x.setLineDash([]);
    }

    // drive balance this step
    const nS = P.series.step.length, tt = Math.min(Math.max(1, P.t), S.T);
    const exc = P.strace[tt * nS] || 0, inh = P.strace[tt * nS + 1] || 0, tot = exc + inh;
    const bx = xr, by = y0 + s + 30, bw = s;
    if (tot > 0 && by + 8 < h) {
      x.fillStyle = gb(0.9); x.fillRect(bx, by, (exc / tot) * bw, 6);
      x.fillStyle = BLUE; x.fillRect(bx + (exc / tot) * bw, by, (inh / tot) * bw, 6);
      x.font = `600 10px ${FH}`; x.textAlign = "left"; x.fillStyle = G[3];
      x.fillText(`EXCITATION ${Math.round((exc / tot) * 100)}% · INHIBITION ${Math.round((inh / tot) * 100)}%`, bx, by - 4);
    }
  }

  function hit(mx, my) {
    for (const b of M.boxes) {
      const c = Math.floor((mx - b.x) / b.cell), r = Math.floor((my - b.y) / b.cell), f = r * b.fw + c;
      if (c >= 0 && c < b.fw && r >= 0 && f < b.count) return [b.pop, b.base + f];
    }
    return null;
  }

  registerView("pcl", popView({
    fig1: "Fig. 01 — event retina → simple → complex · click a cell to probe",
    draw,
    hit,
    tip(mx, my) {
      const u = hit(mx, my);
      if (!u) return null;
      const { k, kc } = shape(), per = u[0] === 2 ? k : kc;
      return [`${S.pops[u[0]].name} · location ${Math.floor(u[1] / per)} · feature ${u[1] % per}`, " · click to probe"];
    },
  }));
}
