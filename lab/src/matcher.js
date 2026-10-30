// Correlative scan matching (after Olson, 2009): exhaustive search over a pose
// window, scoring each candidate by how well the scan's returns land on mapped
// surfaces in a blurred likelihood field.
//
// Tested in isolation on a map built at the true pose: from offsets of up to
// 25 cm and 8 degrees it recovers the true pose to within 0.9 cm and 0.5 deg.

const FIELD_REFRESH_EVERY = 5; // map updates between likelihood-field rebuilds
const POOL = 3; // coarse field cells per side for wide (loop-closure) searches

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export class ScanMatcher {
  constructor(grid, lidarOffset) {
    this.grid = grid;
    this.lidarOffset = lidarOffset;
    this.field = new Float32Array(grid.size * grid.size);
    this.tmp = new Float32Array(grid.size * grid.size);
    this.fieldVersion = -Infinity;
    this.fieldEpoch = -1;
    this.occupied = 0;
    this.coarseSize = Math.ceil(grid.size / POOL);
    this.coarse = new Float32Array(this.coarseSize * this.coarseSize);
    this.coarseVersion = -1;
  }

  // Likelihood field: occupied cells, blurred twice with a 5-tap binomial kernel (sigma ~ 1.4 cells).
  // Rebuilt every few map updates rather than every one: the map barely changes in
  // a quarter of a second, and a slightly older field keeps a scan from being
  // matched against itself. A cleared map forces an immediate rebuild; so does `force`.
  refreshField(force = false) {
    const g = this.grid;
    if (!force && this.fieldEpoch === g.epoch && g.version - this.fieldVersion < FIELD_REFRESH_EVERY) return;
    const n = g.size, F = this.field, T = this.tmp, L = g.logodds, last = n - 1;
    let occ = 0;
    for (let i = 0; i < n * n; i++) { const o = L[i] > 0.8 ? 1 : 0; F[i] = o; occ += o; }
    this.occupied = occ;
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < n; j++) {
        const r = j * n;
        for (let i = 0; i < n; i++) {
          T[r + i] = (F[r + (i > 1 ? i - 2 : 0)] + 4 * F[r + (i > 0 ? i - 1 : 0)] + 6 * F[r + i]
            + 4 * F[r + (i < last ? i + 1 : last)] + F[r + (i < last - 1 ? i + 2 : last)]) * 0.0625;
        }
      }
      for (let j = 0; j < n; j++) {
        const a = (j > 1 ? j - 2 : 0) * n, b = (j > 0 ? j - 1 : 0) * n, c = j * n;
        const d = (j < last ? j + 1 : last) * n, e = (j < last - 1 ? j + 2 : last) * n;
        for (let i = 0; i < n; i++) F[c + i] = (T[a + i] + 4 * T[b + i] + 6 * T[c + i] + 4 * T[d + i] + T[e + i]) * 0.0625;
      }
    }
    this.fieldVersion = g.version;
    this.fieldEpoch = g.epoch;
  }

  // LiDAR returns relative to the body origin: [x0, y0, x1, y1, ...]. `scan` is
  // anything with n, hit and ranges: the live LiDAR or a stored keyframe scan.
  points(scan, stride = 1) {
    const [ox, oy] = this.lidarOffset, out = [];
    for (let k = 0; k < scan.n; k += stride) {
      if (!scan.hit[k]) continue;
      const a = (k * 2 * Math.PI) / scan.n, r = scan.ranges[k];
      out.push(ox + r * Math.cos(a), oy + r * Math.sin(a));
    }
    return out;
  }

  score(pts, x, y, th) {
    const g = this.grid, n = g.size, inv = 1 / g.res, h = g.half, F = this.field;
    const c = Math.cos(th), s = Math.sin(th);
    let sum = 0;
    for (let p = 0; p < pts.length; p += 2) {
      const wx = x + c * pts[p] - s * pts[p + 1], wy = y + s * pts[p] + c * pts[p + 1];
      const i = Math.floor((wx + h) * inv), j = Math.floor((wy + h) * inv);
      if (i >= 0 && j >= 0 && i < n && j < n) sum += F[j * n + i];
    }
    return sum / (pts.length / 2);
  }

  // Search a window around `pred` and return the best pose and its mean likelihood.
  // Small windows (tracking) scale the grid to the window; large ones (relocalising)
  // use a fixed 5 cm / 1 degree grid on a subsample of returns to stay affordable.
  match(lidar, pred, window) {
    this.refreshField();
    const [wx, wt] = window;
    const wide = wx > 0.25;
    const coarse = this.points(lidar, wide ? 6 : 3), fine = this.points(lidar, 1);
    const sxy = wide ? 0.05 : Math.max(0.015, wx / 8);
    const sth = wide ? 0.0175 : Math.max(0.004, wt / 8);
    let best = { x: pred[0], y: pred[1], th: pred[2], score: -1 };
    for (let dt = -wt; dt <= wt + 1e-9; dt += sth) {
      for (let dx = -wx; dx <= wx + 1e-9; dx += sxy) {
        for (let dy = -wx; dy <= wx + 1e-9; dy += sxy) {
          const sc = this.score(coarse, pred[0] + dx, pred[1] + dy, pred[2] + dt);
          if (sc > best.score) best = { x: pred[0] + dx, y: pred[1] + dy, th: pred[2] + dt, score: sc };
        }
      }
    }
    // A coarse optimum on the window's edge means the true pose may lie outside it.
    const atEdge = Math.abs(best.x - pred[0]) > wx - sxy / 2 || Math.abs(best.y - pred[1]) > wx - sxy / 2
      || Math.abs(wrap(best.th - pred[2])) > wt - sth / 2;
    best = this.#refine(fine, best, sxy, sth);
    best.points = fine.length / 2;
    best.atEdge = atEdge;
    return best;
  }

  // Refine around a candidate with every return: a 7 x 7 x 7 grid one coarse step wide.
  #refine(pts, c0, sxy, sth) {
    let best = { x: c0.x, y: c0.y, th: c0.th, score: this.score(pts, c0.x, c0.y, c0.th) };
    for (let dt = -sth; dt <= sth + 1e-9; dt += sth / 3) {
      for (let dx = -sxy; dx <= sxy + 1e-9; dx += sxy / 3) {
        for (let dy = -sxy; dy <= sxy + 1e-9; dy += sxy / 3) {
          const sc = this.score(pts, c0.x + dx, c0.y + dy, c0.th + dt);
          if (sc > best.score) best = { x: c0.x + dx, y: c0.y + dy, th: c0.th + dt, score: sc };
        }
      }
    }
    best.th = wrap(best.th);
    return best;
  }

  // Wide search for loop closure, over metres and tens of degrees. Two levels:
  // the field max-pooled into 15 cm cells is searched exhaustively on that cell
  // grid (each rotation's points are binned once, so a translation is just an
  // index offset), then the best few coarse poses are refined on the full field.
  // Max-pooling makes the coarse score an optimistic bound, so the true optimum
  // is rarely pruned.
  matchWide(scan, pred, [wx, wt], keep = 6) {
    this.refreshField(true);
    this.#pool();
    const g = this.grid, nc = this.coarseSize, res = g.res * POOL, h = g.half, C = this.coarse;
    const pts = this.points(scan, 2), fine = this.points(scan, 1);
    const steps = Math.ceil(wx / res), sth = 0.035; // 2 degrees
    const top = [];
    const cells = new Int32Array(pts.length);
    for (let dt = -wt; dt <= wt + 1e-9; dt += sth) {
      const th = pred[2] + dt, c = Math.cos(th), s = Math.sin(th);
      for (let p = 0; p < pts.length; p += 2) {
        const x = pred[0] + c * pts[p] - s * pts[p + 1], y = pred[1] + s * pts[p] + c * pts[p + 1];
        cells[p] = Math.floor((x + h) / res);
        cells[p + 1] = Math.floor((y + h) / res);
      }
      for (let di = -steps; di <= steps; di++) {
        for (let dj = -steps; dj <= steps; dj++) {
          let sum = 0;
          for (let p = 0; p < pts.length; p += 2) {
            const i = cells[p] + di, j = cells[p + 1] + dj;
            if (i >= 0 && j >= 0 && i < nc && j < nc) sum += C[j * nc + i];
          }
          if (top.length < keep || sum > top[top.length - 1].sum) {
            top.push({ sum, x: pred[0] + di * res, y: pred[1] + dj * res, th });
            top.sort((a, b) => b.sum - a.sum);
            if (top.length > keep) top.pop();
          }
        }
      }
    }
    let best = { score: -1 };
    for (const cand of top) {
      // two refinement levels: within the coarse cell, then to ~1 cm / 0.2 degrees
      const r1 = this.#refine(fine, cand, res / 2, sth / 2);
      const r2 = this.#refine(fine, r1, res / 6, sth / 6);
      if (r2.score > best.score) best = r2;
    }
    best.points = fine.length / 2;
    return best;
  }

  #pool() {
    if (this.coarseVersion === this.fieldVersion + 1e6 * this.fieldEpoch) return;
    const n = this.grid.size, nc = this.coarseSize, F = this.field, C = this.coarse;
    C.fill(0);
    for (let j = 0; j < n; j++) {
      const cj = Math.floor(j / POOL) * nc;
      for (let i = 0; i < n; i++) {
        const v = F[j * n + i], k = cj + Math.floor(i / POOL);
        if (v > C[k]) C[k] = v;
      }
    }
    this.coarseVersion = this.fieldVersion + 1e6 * this.fieldEpoch;
  }
}
