// Pose-graph back end. Keyframe poses are the variables; odometry between
// consecutive keyframes and loop closures between distant ones are the
// constraints. Optimised by Gauss-Newton (Grisetti et al., "A tutorial on
// graph-based SLAM", 2010), solving each step's sparse normal equations with
// conjugate gradients preconditioned by the 3x3 diagonal blocks.

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// Pose b expressed in pose a's frame (all poses [x, y, theta]).
export function relPose(a, b) {
  const c = Math.cos(a[2]), s = Math.sin(a[2]), dx = b[0] - a[0], dy = b[1] - a[1];
  return [c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])];
}

// Pose a followed by the relative motion r.
export function compose(a, r) {
  const c = Math.cos(a[2]), s = Math.sin(a[2]);
  return [a[0] + c * r[0] - s * r[1], a[1] + s * r[0] + c * r[1], wrap(a[2] + r[2])];
}

// 3x3 helpers on flat row-major arrays.
function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-30) return null;
  const k = 1 / det;
  return [A * k, (c * h - b * i) * k, (b * f - c * e) * k, B * k, (a * i - c * g) * k, (c * d - a * f) * k,
    C * k, (b * g - a * h) * k, (a * e - b * d) * k];
}

export class PoseGraph {
  constructor() {
    this.nodes = []; // { pose, fixed, ... }
    this.edges = []; // { i, j, z, info: [3], kind }
  }

  addNode(pose, extra = {}) {
    this.nodes.push({ pose: pose.slice(), fixed: false, ...extra });
    return this.nodes.length - 1;
  }

  // Constraint: node j seen from node i is z, with standard deviations sigma.
  addEdge(i, j, z, sigma, kind) {
    this.edges.push({ i, j, z: z.slice(), info: sigma.map((s) => 1 / (s * s)), kind });
  }

  // Edge residual e = t2v(Z^-1 Xi^-1 Xj) and its Jacobians A = de/dxi, B = de/dxj (3x3, row-major).
  #linearise(edge) {
    const xi = this.nodes[edge.i].pose, xj = this.nodes[edge.j].pose, z = edge.z;
    const e = relPose(z, relPose(xi, xj));
    const ci = Math.cos(xi[2]), si = Math.sin(xi[2]), cz = Math.cos(z[2]), sz = Math.sin(z[2]);
    const dx = xj[0] - xi[0], dy = xj[1] - xi[1];
    // Rz^T Ri^T and Rz^T d(Ri^T)/dtheta (tj - ti)
    const r00 = cz * ci - sz * si, r01 = cz * si + sz * ci;
    const r10 = -sz * ci + cz * -si, r11 = -sz * si + cz * ci;
    const ux = -si * dx + ci * dy, uy = -ci * dx - si * dy; // d(Ri^T)/dtheta (tj - ti)
    const t0 = cz * ux + sz * uy, t1 = -sz * ux + cz * uy;
    const A = [-r00, -r01, t0, -r10, -r11, t1, 0, 0, -1];
    const B = [r00, r01, 0, r10, r11, 0, 0, 0, 1];
    return { e, A, B };
  }

  // Total weighted squared error (chi-square), for reporting convergence.
  chi2() {
    let total = 0;
    for (const edge of this.edges) {
      const { e } = this.#linearise(edge);
      total += e[0] * e[0] * edge.info[0] + e[1] * e[1] * edge.info[1] + e[2] * e[2] * edge.info[2];
    }
    return total;
  }

  optimize(iterations = 6) {
    const n = this.nodes.length;
    if (n < 2) return { iterations: 0 };
    for (let it = 0; it < iterations; it++) {
      // Normal equations H dx = -b, H as 3x3 diagonal blocks plus one off-diagonal block per edge.
      const Hd = new Float64Array(9 * n), b = new Float64Array(3 * n), off = [];
      for (const edge of this.edges) {
        const { e, A, B } = this.#linearise(edge);
        // No robust kernel: a true loop closure's residual is the whole accumulated
        // drift, and a kernel scaled to the edge's noise would ignore exactly the
        // closures that matter. Wrong closures are kept out before they get here.
        const O = edge.info;
        // AtO = A^T Omega, BtO = B^T Omega (3x3)
        const AtO = new Float64Array(9), BtO = new Float64Array(9);
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) { AtO[3 * r + c] = A[3 * c + r] * O[c]; BtO[3 * r + c] = B[3 * c + r] * O[c]; }
        const Hij = new Float64Array(9);
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            let aa = 0, ab = 0, bb = 0;
            for (let k = 0; k < 3; k++) {
              aa += AtO[3 * r + k] * A[3 * k + c];
              ab += AtO[3 * r + k] * B[3 * k + c];
              bb += BtO[3 * r + k] * B[3 * k + c];
            }
            Hd[9 * edge.i + 3 * r + c] += aa;
            Hij[3 * r + c] = ab;
            Hd[9 * edge.j + 3 * r + c] += bb;
          }
          b[3 * edge.i + r] += AtO[3 * r] * e[0] + AtO[3 * r + 1] * e[1] + AtO[3 * r + 2] * e[2];
          b[3 * edge.j + r] += BtO[3 * r] * e[0] + BtO[3 * r + 1] * e[1] + BtO[3 * r + 2] * e[2];
        }
        off.push({ i: edge.i, j: edge.j, H: Hij });
      }
      const fixed = this.nodes.map((nd) => nd.fixed);
      const dx = this.#solve(Hd, off, b, fixed);
      let step = 0;
      for (let k = 0; k < n; k++) {
        if (fixed[k]) continue;
        const p = this.nodes[k].pose;
        p[0] += dx[3 * k]; p[1] += dx[3 * k + 1]; p[2] = wrap(p[2] + dx[3 * k + 2]);
        step = Math.max(step, Math.abs(dx[3 * k]), Math.abs(dx[3 * k + 1]));
      }
      if (step < 1e-4) return { iterations: it + 1 };
    }
    return { iterations };
  }

  // Preconditioned conjugate gradients on H dx = -b, with fixed nodes held at zero.
  // The preconditioner is H restricted to the odometry chain (every diagonal block
  // plus the blocks between consecutive keyframes), which is block-tridiagonal and
  // solved exactly by block elimination. PCG then only has to account for the loop
  // closures, so it needs a handful of iterations instead of one per variable
  // (subgraph preconditioning, Dellaert et al. 2010).
  #solve(Hd, off, b, fixed) {
    const n = fixed.length, N = 3 * n;
    const blk = (A, k) => A.subarray(9 * k, 9 * k + 9);
    const mm = (A, B) => { const C = new Float64Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]; return C; };
    const tr = (A) => Float64Array.of(A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]);
    const mv = (A, x, o) => [A[0] * x[o] + A[1] * x[o + 1] + A[2] * x[o + 2], A[3] * x[o] + A[4] * x[o + 1] + A[5] * x[o + 2], A[6] * x[o] + A[7] * x[o + 1] + A[8] * x[o + 2]];

    // C[k] = H(k-1, k): the chain block linking keyframe k to the one before it.
    const C = new Array(n).fill(null);
    for (const { i, j, H } of off) {
      const k = j === i + 1 ? j : i === j + 1 ? i : -1;
      if (k < 0) continue;
      const Hk = j === i + 1 ? H : tr(H);
      if (C[k]) for (let a = 0; a < 9; a++) C[k][a] += Hk[a]; else C[k] = Float64Array.from(Hk);
    }
    const linked = (k) => k > 0 && C[k] && !fixed[k] && !fixed[k - 1];
    // Block LDL^T of the chain: Dp_k = D_k - C_k^T Dp_{k-1}^-1 C_k.
    const Dinv = new Array(n).fill(null), W = new Array(n).fill(null); // W_k = C_k^T Dp_{k-1}^-1
    for (let k = 0; k < n; k++) {
      if (fixed[k]) continue;
      const D = Float64Array.from(blk(Hd, k));
      if (linked(k)) {
        W[k] = mm(tr(C[k]), Dinv[k - 1]);
        const S = mm(W[k], C[k]);
        for (let a = 0; a < 9; a++) D[a] -= S[a];
      }
      Dinv[k] = Float64Array.from(inv3(Array.from(D)) || [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }
    const precond = (r, z) => {
      const y = Float64Array.from(r);
      for (let k = 0; k < n; k++) {
        if (fixed[k]) { y[3 * k] = y[3 * k + 1] = y[3 * k + 2] = 0; continue; }
        if (linked(k)) { const t = mv(W[k], y, 3 * (k - 1)); y[3 * k] -= t[0]; y[3 * k + 1] -= t[1]; y[3 * k + 2] -= t[2]; }
      }
      for (let k = n - 1; k >= 0; k--) {
        if (fixed[k]) { z[3 * k] = z[3 * k + 1] = z[3 * k + 2] = 0; continue; }
        let t0 = y[3 * k], t1 = y[3 * k + 1], t2 = y[3 * k + 2];
        if (k + 1 < n && linked(k + 1)) { const t = mv(C[k + 1], z, 3 * (k + 1)); t0 -= t[0]; t1 -= t[1]; t2 -= t[2]; }
        const D = Dinv[k];
        z[3 * k] = D[0] * t0 + D[1] * t1 + D[2] * t2;
        z[3 * k + 1] = D[3] * t0 + D[4] * t1 + D[5] * t2;
        z[3 * k + 2] = D[6] * t0 + D[7] * t1 + D[8] * t2;
      }
    };
    const mulH = (x, y) => {
      y.fill(0);
      for (let k = 0; k < n; k++) {
        if (fixed[k]) continue;
        const t = mv(blk(Hd, k), x, 3 * k);
        y[3 * k] = t[0]; y[3 * k + 1] = t[1]; y[3 * k + 2] = t[2];
      }
      for (const { i, j, H } of off) {
        if (fixed[i] || fixed[j]) continue;
        for (let r = 0; r < 3; r++) {
          y[3 * i + r] += H[3 * r] * x[3 * j] + H[3 * r + 1] * x[3 * j + 1] + H[3 * r + 2] * x[3 * j + 2];
          y[3 * j + r] += H[r] * x[3 * i] + H[3 + r] * x[3 * i + 1] + H[6 + r] * x[3 * i + 2];
        }
      }
    };
    const x = new Float64Array(N), r = new Float64Array(N), z = new Float64Array(N), p = new Float64Array(N), q = new Float64Array(N);
    for (let k = 0; k < n; k++) if (!fixed[k]) for (let a = 0; a < 3; a++) r[3 * k + a] = -b[3 * k + a];
    precond(r, z);
    p.set(z);
    let rz = 0, r0 = 0;
    for (let k = 0; k < N; k++) { rz += r[k] * z[k]; r0 += r[k] * r[k]; }
    if (r0 === 0) return x;
    for (let it = 0; it < 200; it++) {
      this.cgIterations = it + 1;
      mulH(p, q);
      let pq = 0;
      for (let k = 0; k < N; k++) pq += p[k] * q[k];
      if (pq <= 0) break;
      const alpha = rz / pq;
      let rr = 0;
      for (let k = 0; k < N; k++) { x[k] += alpha * p[k]; r[k] -= alpha * q[k]; rr += r[k] * r[k]; }
      if (rr < 1e-16 * r0) break;
      precond(r, z);
      let rzNew = 0;
      for (let k = 0; k < N; k++) rzNew += r[k] * z[k];
      const beta = rzNew / rz;
      rz = rzNew;
      for (let k = 0; k < N; k++) p[k] = z[k] + beta * p[k];
    }
    return x;
  }
}
