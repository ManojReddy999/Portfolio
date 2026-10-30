// 2D SLAM front end: noisy wheel/gyro odometry in an EKF that also estimates
// the gyro bias and wheel scale error, corrected by correlative scan matching
// against the occupancy map being built.
//
// The simulated encoders and gyro are perfect (the gyro is exactly the previous
// physics step's yaw rate), so all error here comes from the synthetic noise
// below: a per-session wheel scale error and gyro bias, plus per-sample noise.
// That is the error a real car has; it is injected, not emergent.
//
// Why the biases are states: they are systematic, so their effect grows with
// time, not its square root. A white-noise model cannot represent that; the
// filter becomes overconfident, its gain toward scan matches collapses, and
// the 3-sigma search window shrinks around a pose that has already drifted.

const REAR_AXLE_TO_ORIGIN = 0.158; // body origin sits this far ahead of the rear axle (buddy.xml)
const FIELD_REFRESH_EVERY = 5; // map updates between likelihood-field rebuilds
const BIAS_LIMIT = 0.1; // rad/s (5.7 deg/s); injected bias is at most 0.06 at 3x noise
const SCALE_LIMIT = 0.15; // injected wheel scale error is at most 9% at 3x noise
// Effective rolling radius, calibrated from the simulator: least squares of true
// forward speed on rear-encoder rate over 60 s of driving (R^2 = 0.998). 7.2%
// above the nominal 0.0488 m because the collision cylinders roll on their rim.
export const WHEEL_RADIUS = 0.05234;

function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// ---- small dense matrix helpers (arrays of rows) ---------------------------

const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const eye = (n) => { const I = zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; };
function mul(A, B) {
  const C = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i++) for (let k = 0; k < B.length; k++) {
    const a = A[i][k];
    if (a !== 0) for (let j = 0; j < B[0].length; j++) C[i][j] += a * B[k][j];
  }
  return C;
}
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
function inv3(M) {
  const [[a, b, c], [d, e, f], [g, h, i]] = M;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-18) return null;
  const k = 1 / det;
  return [[A * k, -(b * i - c * h) * k, (b * f - c * e) * k],
    [B * k, (a * i - c * g) * k, -(a * f - c * d) * k],
    [C * k, -(a * h - b * g) * k, (a * e - b * d) * k]];
}

// ---------------------------------------------------------------------------
// Odometry: integrates noisy encoder and gyro readings at the physics rate.
// Returns the RAW (uncorrected) motion since the last call, in the body frame
// at its start; the filter applies its bias and scale estimates.

export class Odometry {
  constructor(sim, rng, level = 1) {
    const m = sim.model, mj = sim.mujoco;
    const adr = (name) => m.sensor_adr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR.value, name)];
    this.sim = sim;
    this.adr = { gyro: adr('car/buddy_gyro') + 2, bl: adr('car/buddy_wheel_bl_vel'), br: adr('car/buddy_wheel_br_vel') };
    this.rng = rng;
    // Session errors the filter must discover for itself.
    const sign = () => (rng() < 0.5 ? -1 : 1);
    this.baseScale = sign() * (0.01 + 0.02 * rng());
    this.baseBias = sign() * (0.008 + 0.012 * rng());
    this.setLevel(level);
    this.reset();
  }

  setLevel(level) {
    this.level = level;
    this.scaleError = level * this.baseScale; // true value, for display only
    this.gyroBias = level * this.baseBias;
  }

  reset() { this.dx = 0; this.dy = 0; this.dth = 0; this.ds = 0; this.T = 0; }

  // Called after every physics step.
  sample(dt) {
    const sd = this.sim.data.sensordata, k = this.level, rng = this.rng;
    const wheel = 0.5 * (sd[this.adr.bl] + sd[this.adr.br]) + k * 0.25 * gaussian(rng);
    const v = WHEEL_RADIUS * (1 + this.scaleError) * wheel;
    const w = sd[this.adr.gyro] + this.gyroBias + k * 0.03 * gaussian(rng);
    // Body-origin velocity in the body frame: rear-axle speed forward, plus the
    // sideways swing of a point ahead of the rear axle while turning.
    const vx = v, vy = w * REAR_AXLE_TO_ORIGIN;
    const c = Math.cos(this.dth), s = Math.sin(this.dth);
    this.dx += (c * vx - s * vy) * dt;
    this.dy += (s * vx + c * vy) * dt;
    this.dth += w * dt;
    this.ds += Math.abs(v) * dt;
    this.T += dt;
  }

  take() {
    const d = { dx: this.dx, dy: this.dy, dth: this.dth, ds: this.ds, T: this.T };
    this.reset();
    return d;
  }
}

// ---------------------------------------------------------------------------
// EKF over [x, y, theta, gyro bias (rad/s), wheel scale error].

export class PoseEKF {
  constructor() { this.reset(0, 0, 0, 1); }

  reset(x, y, th, level) {
    const L = Math.max(level, 0.2);
    this.x = [x, y, th, 0, 0];
    this.P = zeros(5, 5);
    this.P[0][0] = this.P[1][1] = 0.01 ** 2;
    this.P[2][2] = 0.005 ** 2;
    this.P[3][3] = (0.025 * L) ** 2; // prior on the gyro bias
    this.P[4][4] = (0.035 * L) ** 2; // prior on the wheel scale error
  }

  // Noise level changed: the sensor errors changed, so reopen the bias priors.
  reopenBiases(level) {
    const L = Math.max(level, 0.2);
    this.P[3][3] = Math.max(this.P[3][3], (0.025 * L) ** 2);
    this.P[4][4] = Math.max(this.P[4][4], (0.035 * L) ** 2);
  }

  // `slipping`: the wheels have been seen spinning without the car moving, so
  // the encoders' distance is only an upper bound on the real motion.
  predict(d, level, slipping = false) {
    const [x, y, th, b, s] = this.x, c = Math.cos(th), sn = Math.sin(th);
    const dxc = d.dx * (1 - s), dyc = d.dy; // wheel-scale correction on the along-track motion
    const dthc = d.dth - b * d.T; // gyro-bias correction
    this.x = [x + c * dxc - sn * dyc, y + sn * dxc + c * dyc, wrap(th + dthc), b, s];

    const F = eye(5);
    F[0][2] = -sn * dxc - c * dyc; F[1][2] = c * dxc - sn * dyc;
    // While the wheels slip, the encoder distance is not real motion, so the wheel
    // scale has no bearing on it. Coupling them would let a slip leak into the
    // scale estimate through the covariance once the slip ends.
    if (!slipping) { F[0][4] = -c * d.dx; F[1][4] = -sn * d.dx; }
    F[2][3] = -d.T;

    // White noise only: per-sample sensor noise plus unmodelled slip.
    const k = level;
    const sa = slipping ? 0.8 * d.ds + 0.002 : 0.0003 * k + 0.006 * d.ds + 0.0002; // along track
    const sl = 0.003 * d.ds + 0.0002; // lateral
    const st = 0.0005 * k + 0.005 * Math.abs(d.dth) + 0.0001;
    const P = mul(mul(F, this.P), transpose(F));
    P[0][0] += c * c * sa * sa + sn * sn * sl * sl;
    P[1][1] += sn * sn * sa * sa + c * c * sl * sl;
    P[0][1] += c * sn * (sa * sa - sl * sl); P[1][0] = P[0][1];
    P[2][2] += st * st;
    P[3][3] += 1e-5 ** 2; // slow bias random walk
    P[4][4] += 2e-5 ** 2;
    this.P = P;
  }

  // Squared Mahalanobis distance of a pose measurement from the prediction.
  mahalanobis2(z, R) {
    const P = this.P;
    const Si = inv3([0, 1, 2].map((i) => [0, 1, 2].map((j) => P[i][j] + (i === j ? R[i] : 0))));
    if (!Si) return Infinity;
    const r = [z[0] - this.x[0], z[1] - this.x[1], wrap(z[2] - this.x[2])];
    let d2 = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) d2 += r[i] * Si[i][j] * r[j];
    return d2;
  }

  // Innovation of a pose measurement along the heading (m): positive = ahead of the prediction.
  alongTrack(z) {
    const th = this.x[2];
    return (z[0] - this.x[0]) * Math.cos(th) + (z[1] - this.x[1]) * Math.sin(th);
  }

  // Slip hypothesis: the last `dist` metres the encoders reported may not have
  // happened. Add that much variance along the heading.
  inflateAlongTrack(dist) {
    const th = this.x[2], c = Math.cos(th), s = Math.sin(th), v = dist * dist;
    this.P[0][0] += c * c * v; this.P[1][1] += s * s * v;
    this.P[0][1] += c * s * v; this.P[1][0] += c * s * v;
  }

  // Pose measurement from scan matching: H = [I3 0].
  // `freezeCalibration` keeps a slip episode from being learned as a wheel-scale or gyro error.
  update(z, R, freezeCalibration = false) {
    const P = this.P;
    const S = [0, 1, 2].map((i) => [0, 1, 2].map((j) => P[i][j] + (i === j ? R[i] : 0)));
    const Si = inv3(S);
    if (!Si) return false;
    const PHt = P.map((row) => row.slice(0, 3)); // 5x3
    const K = mul(PHt, Si); // 5x3
    if (freezeCalibration) K[3] = [0, 0, 0], K[4] = [0, 0, 0];
    const r = [z[0] - this.x[0], z[1] - this.x[1], wrap(z[2] - this.x[2])];
    this.x = this.x.map((v, i) => v + K[i][0] * r[0] + K[i][1] * r[1] + K[i][2] * r[2]);
    this.x[2] = wrap(this.x[2]);
    // Physical bounds on the calibration states, wider than any injected error.
    this.x[3] = Math.max(-BIAS_LIMIT, Math.min(BIAS_LIMIT, this.x[3]));
    this.x[4] = Math.max(-SCALE_LIMIT, Math.min(SCALE_LIMIT, this.x[4]));
    // P = (I - K H) P
    const KH = zeros(5, 5);
    for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) KH[i][j] = K[i][j];
    const IKH = eye(5).map((row, i) => row.map((v, j) => v - KH[i][j]));
    this.P = mul(IKH, P);
    return true;
  }

  sigma() { return [Math.sqrt(this.P[0][0]), Math.sqrt(this.P[1][1]), Math.sqrt(this.P[2][2])]; }

  // Row-major x-y-theta covariance block, for drawing.
  poseCov() { const P = this.P; return [P[0][0], P[0][1], P[0][2], P[1][0], P[1][1], P[1][2], P[2][0], P[2][1], P[2][2]]; }
}

// ---------------------------------------------------------------------------
// Correlative scan matcher (after Olson, 2009): exhaustive search over a pose
// window, scoring each candidate by how well the scan's returns land on mapped
// surfaces in a blurred likelihood field. The window is the EKF's 3-sigma
// uncertainty, so the filter tells the matcher where to look.
//
// Tested in isolation on a map built at the true pose: from offsets of up to
// 25 cm and 8 degrees it recovers the true pose to within 0.9 cm and 0.5 deg.

export class ScanMatcher {
  constructor(grid, lidarOffset) {
    this.grid = grid;
    this.lidarOffset = lidarOffset;
    this.field = new Float32Array(grid.size * grid.size);
    this.tmp = new Float32Array(grid.size * grid.size);
    this.fieldVersion = -Infinity;
    this.fieldEpoch = -1;
    this.occupied = 0;
  }

  // Likelihood field: occupied cells, blurred twice with a 5-tap binomial kernel (sigma ~ 1.4 cells).
  // Rebuilt every few map updates rather than every one: the map barely changes in
  // a quarter of a second, and a slightly older field keeps a scan from being
  // matched against itself. A cleared map forces an immediate rebuild.
  refreshField() {
    const g = this.grid;
    if (this.fieldEpoch === g.epoch && g.version - this.fieldVersion < FIELD_REFRESH_EVERY) return;
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

  // LiDAR returns relative to the body origin: [x0, y0, x1, y1, ...].
  points(lidar, stride = 1) {
    const [ox, oy] = this.lidarOffset, out = [];
    for (let k = 0; k < lidar.n; k += stride) {
      if (!lidar.hit[k]) continue;
      const a = (k * 2 * Math.PI) / lidar.n, r = lidar.ranges[k];
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
    // Refine around the coarse optimum with every return.
    const c0 = { ...best };
    best.score = this.score(fine, c0.x, c0.y, c0.th);
    for (let dt = -sth; dt <= sth + 1e-9; dt += sth / 3) {
      for (let dx = -sxy; dx <= sxy + 1e-9; dx += sxy / 3) {
        for (let dy = -sxy; dy <= sxy + 1e-9; dy += sxy / 3) {
          const sc = this.score(fine, c0.x + dx, c0.y + dy, c0.th + dt);
          if (sc > best.score) best = { x: c0.x + dx, y: c0.y + dy, th: c0.th + dt, score: sc };
        }
      }
    }
    best.th = wrap(best.th);
    best.points = fine.length / 2;
    best.atEdge = atEdge;
    return best;
  }
}

// ---------------------------------------------------------------------------
// The pieces wired together.

const MATCH_EVERY = 2; // control steps between scan matches while tracking (10 Hz)
const RELOCALISE_EVERY = 10; // and while searching for a lost pose (2 Hz)
const MAX_WINDOW = [1.0, 0.35]; // metres, radians (20 degrees)
const MIN_POINTS = 50;
const MIN_OCCUPIED = 250; // cells; don't match against a map that is barely there
const MIN_SCORE = 0.2; // mean likelihood per return to accept a match (about 0.32 when aligned)
const R_MATCH = [0.012 ** 2, 0.012 ** 2, 0.006 ** 2]; // from the isolated matcher test
const GATE = 16.27; // chi-square, 3 dof, 99.9%: reject matches the filter finds implausible

export class Slam {
  constructor(sim, grid, rng, level, lidarOffset) {
    this.odom = new Odometry(sim, rng, level);
    this.ekf = new PoseEKF();
    this.matcher = new ScanMatcher(grid, lidarOffset);
    this.lidarOffset = lidarOffset;
    this.matching = true;
    this.level = level;
    this.steps = 0;
    this.state = 'tracking';
    this.rejected = 0;
    this.lastScore = 0;
    this.encSinceMatch = 0; // encoder distance since the last accepted match
    this.encAlongSinceMatch = 0; // the same, signed along the heading (negative when reversing)
    this.slipUntil = 0; // control step until which the wheels are treated as slipping
    this.slipEvents = 0;
  }

  get slipping() { return this.steps < this.slipUntil; }

  setLevel(level) {
    this.level = level;
    this.odom.setLevel(level);
    this.ekf.reopenBiases(level);
  }

  // Known (re)initialisation, e.g. at the start or after a respawn.
  reinit(x, y, th) {
    const [b, s] = [this.ekf.x[3], this.ekf.x[4]];
    const [vb, vs] = [this.ekf.P[3][3], this.ekf.P[4][4]];
    const fresh = this.steps === 0;
    this.ekf.reset(x, y, th, this.level);
    if (!fresh) {
      // Keep what it learned about its sensors, and how sure it is of it; the new
      // pose is independent of both.
      this.ekf.x[3] = b; this.ekf.x[4] = s;
      this.ekf.P[3][3] = vb; this.ekf.P[4][4] = vs;
    }
    this.odom.reset();
    this.rejected = 0;
    this.encSinceMatch = 0;
    this.encAlongSinceMatch = 0;
    this.slipUntil = 0;
    this.state = this.matching ? 'tracking' : 'dead reckoning';
  }

  // Whether scans should be written into the map. Without localisation they
  // would corrupt it: later matches would then confirm the drift (a false lock).
  get localised() { return this.matching && this.state === 'tracking'; }

  // After the physics step: propagate, then correct against the map if enabled.
  // Returns true on the first accepted match after a period without localisation.
  update(lidar) {
    const d = this.odom.take();
    const slipping = this.slipping;
    this.ekf.predict(d, this.level, slipping);
    this.encSinceMatch += d.ds;
    this.encAlongSinceMatch += d.dx;
    this.steps++;
    if (!this.matching) { this.state = 'dead reckoning'; return false; }
    if (this.steps % (this.state === 'tracking' ? MATCH_EVERY : RELOCALISE_EVERY)) return false;

    const [sx, sy, st] = this.ekf.sigma();
    let window = [
      Math.min(MAX_WINDOW[0], Math.max(0.04, 3 * Math.max(sx, sy))),
      Math.min(MAX_WINDOW[1], Math.max(0.02, 3 * st)),
    ];
    this.matcher.refreshField();
    let hits = 0;
    for (let k = 0; k < lidar.n; k++) hits += lidar.hit[k];
    const wasUnlocalised = this.state !== 'tracking';
    if (hits < MIN_POINTS || this.matcher.occupied < MIN_OCCUPIED) {
      if (this.state === 'dead reckoning') this.state = 'relocalising';
      return false;
    }

    let m = this.matcher.match(lidar, this.ekf.x, window);
    // Best pose on the window's edge: the truth may lie outside it, so look wider.
    for (let k = 0; k < 2 && m.atEdge && window[0] < MAX_WINDOW[0]; k++) {
      window = [Math.min(MAX_WINDOW[0], 2 * window[0]), Math.min(MAX_WINDOW[1], 2 * window[1])];
      m = this.matcher.match(lidar, this.ekf.x, window);
    }
    this.lastScore = m.score;
    const z = [m.x, m.y, m.th];
    let accept = m.score >= MIN_SCORE && this.ekf.mahalanobis2(z, R_MATCH) < GATE;

    // How much of the motion the encoders claimed along the heading the scan
    // contradicts: positive when the car went less far than the wheels said.
    const enc = this.encAlongSinceMatch;
    const shortfall = Math.sign(enc) * -this.ekf.alongTrack(z);

    // A good match the filter calls implausible, where the car went less far than
    // the encoders said: test the hypothesis that the wheels spun and that motion
    // never happened. Accept it only if that makes it plausible.
    let slip = false;
    if (!accept && m.score >= MIN_SCORE && shortfall > 0.03 && Math.abs(enc) > 0.03) {
      const saved = this.ekf.P.map((row) => row.slice());
      this.ekf.inflateAlongTrack(Math.abs(enc));
      if (this.ekf.mahalanobis2(z, R_MATCH) < GATE) accept = slip = true;
      else this.ekf.P = saved;
    }
    // Even when the match was plausible (the filter may already be allowing for
    // slip), the scan showing under half the claimed travel IS slip: keep the flag
    // up so a spinning wheel is never learned as a wheel-scale error.
    if (accept && Math.abs(enc) > 0.03 && shortfall > 0.5 * Math.abs(enc)) slip = true;

    // Kept for inspection: what the last match saw and decided.
    this.lastMatch = {
      score: m.score, atEdge: m.atEdge, window: window.slice(), shortfall, enc,
      d2: this.ekf.mahalanobis2(z, R_MATCH), accept, slip,
    };

    if (accept) {
      if (slip) { this.slipUntil = this.steps + 10; this.slipEvents++; }
      this.ekf.update(z, R_MATCH, slip || slipping); // don't learn calibration from a slip
      this.encSinceMatch = 0;
      this.encAlongSinceMatch = 0;
      this.rejected = 0;
      this.state = 'tracking';
      return wasUnlocalised;
    }
    this.rejected++;
    if (this.rejected > 10) this.state = 'lost';
    else if (this.state === 'dead reckoning') this.state = 'relocalising';
    return false;
  }

  // Estimated sensor pose, for building the map.
  sensorPose() {
    const [x, y, th] = this.ekf.x, [ox, oy] = this.lidarOffset;
    const c = Math.cos(th), s = Math.sin(th);
    return { x: x + c * ox - s * oy, y: y + s * ox + c * oy, yaw: th };
  }

  pose() { return this.ekf.x.slice(0, 3); }
}
