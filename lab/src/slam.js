// 2D SLAM. Front end: noisy wheel/gyro odometry in an EKF that also estimates
// the gyro bias and wheel scale error, corrected by correlative scan matching
// against a local map of the recent route. Back end: keyframes in a pose graph,
// with loop closures when the car comes back to a place it mapped long ago; the
// graph is then re-optimised and the map rebuilt from the corrected keyframes.
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

import { OccupancyGrid } from './mapping.js';
import { ScanMatcher } from './matcher.js';
import { PoseGraph, relPose, compose } from './graph.js';
import { LoopCloser, LOOP_SIGMA } from './loop.js';

const REAR_AXLE_TO_ORIGIN = 0.158; // body origin sits this far ahead of the rear axle (buddy.xml)
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
// The pieces wired together.

const MATCH_EVERY = 2; // control steps between scan matches while tracking (10 Hz)
const RELOCALISE_EVERY = 10; // and while searching for a lost pose (2 Hz)
const MAX_WINDOW = [1.0, 0.35]; // metres, radians (20 degrees)
const MIN_POINTS = 50;
const MIN_OCCUPIED = 250; // cells; don't match against a map that is barely there
const MIN_SCORE = 0.2; // mean likelihood per return to accept a match (about 0.32 when aligned)
const R_MATCH = [0.012 ** 2, 0.012 ** 2, 0.006 ** 2]; // from the isolated matcher test
const GATE = 16.27; // chi-square, 3 dof, 99.9%: reject matches the filter finds implausible
const KF_DIST = 0.4; // m of travel between keyframes
const KF_TURN = 0.26; // rad (15 degrees) of turning between keyframes
const LOCAL_KEYFRAMES = 40; // the front end's local map: the last ~16 m of route
const LOCAL_REBUILD_EVERY = 20; // keyframes between local-map rebuilds
const LOOP_SEARCH_EVERY = 2; // keyframes between loop-closure searches
export const TUNING = { odomScale: 1, searchEvery: LOOP_SEARCH_EVERY }; // benchmark experiments only
const SMALL_CORRECTION = [0.03, 0.017]; // m, rad (1 degree): below this a closure only confirms the map
// Odometry-edge noise per keyframe step, for the front end's scan-matched motion.
const odomSigma = (d, dth) => [0.01 + 0.03 * d, 0.01 + 0.03 * d, 0.005 + 0.03 * Math.abs(dth) + 0.01 * d];

export class Slam {
  // `grid` is the global map: drawn on screen, and used to relocalise after the
  // car has been driving blind. Tracking matches against a local map of the
  // recent route instead, so drift really does build up and loop closure has to
  // remove it. `truth()` (the simulator's pose) is recorded per keyframe for
  // evaluation only; nothing here reads it.
  constructor(sim, grid, rng, level, lidarOffset, { loopClosure = true } = {}) {
    this.odom = new Odometry(sim, rng, level);
    this.ekf = new PoseEKF();
    this.global = grid;
    this.local = new OccupancyGrid({ half: grid.half, res: grid.res });
    this.matcher = new ScanMatcher(this.local, lidarOffset);
    this.globalMatcher = new ScanMatcher(grid, lidarOffset);
    this.graph = new PoseGraph();
    this.closer = new LoopCloser(this.graph, lidarOffset, grid);
    this.loopClosure = loopClosure;
    this.loopEvents = [];
    this.truth = () => sim.carPose();
    this.segment = 0;
    this.lastKf = -1; // the newest keyframe of the current segment
    this.kfSinceRebuild = 0;
    this.sinceClosure = 0; // metres of route since the estimate was last pinned down
    this.blindSince = false; // drove without localisation since the last keyframe
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

  // Known (re)initialisation, e.g. at the start or after a respawn. Given the
  // scan taken there, it anchors a new segment of the pose graph.
  reinit(x, y, th, lidar = null) {
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
    this.segment++;
    this.lastKf = -1;
    this.sinceClosure = 0;
    this.blindSince = false;
    this.closer.reset();
    this.local.clear();
    if (lidar) {
      this.#addKeyframe(lidar, true);
      this.local.integrate(lidar, this.sensorPose());
    }
  }

  // Forget the map and the graph (the lab's "restart map").
  resetMap() {
    this.graph = new PoseGraph();
    this.closer = new LoopCloser(this.graph, this.lidarOffset, this.global);
    this.lastKf = -1;
    this.loopEvents = [];
  }

  // ---- keyframes, local map and loop closure --------------------------------

  #addKeyframe(lidar, anchor = false) {
    const pose = this.pose();
    const k = this.graph.addNode(pose, {
      segment: this.segment,
      scan: { n: lidar.n, ranges: Float32Array.from(lidar.ranges), hit: Uint8Array.from(lidar.hit) },
      truth: this.truth(), // evaluation only
    });
    if (anchor) {
      this.graph.nodes[k].fixed = true; // a known pose: the start, or a respawn
    } else if (this.lastKf >= 0) {
      const z = relPose(this.lastKfFront, pose);
      // After driving blind, the step back to the last keyframe is only dead reckoning.
      const sigma = this.blindSince ? [0.3, 0.3, 0.15] : odomSigma(Math.hypot(z[0], z[1]), z[2]).map((v) => v * TUNING.odomScale);
      this.graph.addEdge(this.lastKf, k, z, sigma, 'odom');
      this.sinceClosure += Math.hypot(z[0], z[1]);
    }
    this.blindSince = false;
    this.lastKf = k;
    this.lastKfFront = pose;
    this.global.integrate(lidar, this.sensorPose());
    if (++this.kfSinceRebuild >= LOCAL_REBUILD_EVERY) this.#rebuildLocal();
    if (!anchor && this.loopClosure && k % TUNING.searchEvery === 0) {
      const radius = Math.min(2.0, 0.6 + 0.03 * this.sinceClosure);
      const pair = this.closer.attempt(k, radius);
      if (pair) this.#close(pair);
    }
  }

  // The local map: the last LOCAL_KEYFRAMES keyframes of this segment, at their graph poses.
  #rebuildLocal() {
    this.kfSinceRebuild = 0;
    this.local.clear();
    const nodes = this.graph.nodes;
    for (let c = this.lastKf, used = 0; c >= 0 && used < LOCAL_KEYFRAMES; c--, used++) {
      if (nodes[c].segment !== this.segment) break;
      this.local.integrate(nodes[c].scan, this.#sensorAt(nodes[c].pose));
    }
  }

  #sensorAt(pose) {
    const [ox, oy] = this.lidarOffset, c = Math.cos(pose[2]), s = Math.sin(pose[2]);
    return { x: pose[0] + c * ox - s * oy, y: pose[1] + s * ox + c * oy, yaw: pose[2] };
  }

  // A confirmed loop: add both constraints, re-optimise, carry the live estimate
  // along with its keyframe, and rebuild both maps from the corrected keyframes.
  #close(pair) {
    for (const h of pair) this.graph.addEdge(h.c, h.k, h.z, LOOP_SIGMA, 'loop');
    // Once the map is consistent, most revisits only confirm it. Keep those
    // constraints, but skip the re-optimisation and map rebuild until a closure
    // actually implies a correction.
    const h = pair[1], off = Math.hypot(h.found[0] - h.pose[0], h.found[1] - h.pose[1]);
    const turn = Math.abs(Math.atan2(Math.sin(h.found[2] - h.pose[2]), Math.cos(h.found[2] - h.pose[2])));
    if (off < SMALL_CORRECTION[0] && turn < SMALL_CORRECTION[1]) {
      this.sinceClosure = 0;
      this.loopEvents.push({ from: this.graph.nodes[h.c].pose.slice(), to: h.pose.slice(), shift: 0, path: null, ms: 0, iterations: 0 });
      return;
    }
    const before = this.graph.nodes[this.lastKf].pose.slice();
    const t0 = performance.now();
    const res = this.graph.optimize(8);
    const after = this.graph.nodes[this.lastKf].pose;
    const moved = compose(after, relPose(before, this.ekf.x.slice(0, 3)));
    this.ekf.x[0] = moved[0]; this.ekf.x[1] = moved[1]; this.ekf.x[2] = moved[2];
    this.lastKfFront = after.slice();
    this.sinceClosure = 0;
    this.global.clear();
    for (const nd of this.graph.nodes) this.global.integrate(nd.scan, this.#sensorAt(nd.pose));
    this.#rebuildLocal();
    this.loopEvents.push({
      from: this.graph.nodes[pair[1].c].pose.slice(), to: after.slice(),
      shift: Math.hypot(after[0] - before[0], after[1] - before[1]),
      path: this.graph.nodes.map((nd) => [nd.pose[0], nd.pose[1], nd.segment]),
      ms: performance.now() - t0, iterations: res.iterations,
    });
  }

  takeLoopEvents() { const e = this.loopEvents; this.loopEvents = []; return e; }

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
    if (!this.matching) { this.state = 'dead reckoning'; this.blindSince = true; return false; }
    if (this.steps % (this.state === 'tracking' ? MATCH_EVERY : RELOCALISE_EVERY)) {
      this.#afterStep(lidar);
      return false;
    }
    // Tracking matches the recent route; finding a lost pose needs the whole map.
    const matcher = this.state === 'tracking' ? this.matcher : this.globalMatcher;

    const [sx, sy, st] = this.ekf.sigma();
    let window = [
      Math.min(MAX_WINDOW[0], Math.max(0.04, 3 * Math.max(sx, sy))),
      Math.min(MAX_WINDOW[1], Math.max(0.02, 3 * st)),
    ];
    matcher.refreshField();
    let hits = 0;
    for (let k = 0; k < lidar.n; k++) hits += lidar.hit[k];
    const wasUnlocalised = this.state !== 'tracking';
    if (hits < MIN_POINTS || matcher.occupied < MIN_OCCUPIED) {
      if (this.state === 'dead reckoning') this.state = 'relocalising';
      this.#afterStep(lidar);
      return false;
    }

    let m = matcher.match(lidar, this.ekf.x, window);
    // Best pose on the window's edge: the truth may lie outside it, so look wider.
    for (let k = 0; k < 2 && m.atEdge && window[0] < MAX_WINDOW[0]; k++) {
      window = [Math.min(MAX_WINDOW[0], 2 * window[0]), Math.min(MAX_WINDOW[1], 2 * window[1])];
      m = matcher.match(lidar, this.ekf.x, window);
    }
    this.lastScore = m.score;
    const z = [m.x, m.y, m.th];
    let accept = m.score >= MIN_SCORE && this.ekf.mahalanobis2(z, R_MATCH) < GATE;

    // How much of the motion the encoders claimed along the heading the scan
    // contradicts: positive when the car went less far than the wheels said.
    const enc = this.encAlongSinceMatch;
    const shortfall = Math.sign(enc) * -this.ekf.alongTrack(z);
    // Slip can make the car go less far than the wheels claim, never backwards:
    // a match that puts it behind where it started (when the wheels say it went
    // forward) is a wrong match, however plausible an inflated covariance makes it.
    if (Math.abs(enc) > 0.03 && shortfall > Math.abs(enc) + 0.05) accept = false;

    // A good match the filter calls implausible, where the car went less far than
    // the encoders said: test the hypothesis that the wheels spun and that motion
    // never happened. Accept it only if that makes it plausible.
    let slip = false;
    if (!accept && m.score >= MIN_SCORE && shortfall > 0.03 && Math.abs(enc) > 0.03 && shortfall <= Math.abs(enc) + 0.05) {
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
      if (wasUnlocalised) this.#rebuildLocal(); // the local map may be far behind the car
      this.#afterStep(lidar);
      return wasUnlocalised;
    }
    this.rejected++;
    if (this.rejected > 10) this.state = 'lost';
    else if (this.state === 'dead reckoning') this.state = 'relocalising';
    if (this.state !== 'tracking') this.blindSince = true;
    return false;
  }

  // Every control step while localised: extend the local map, and add a
  // keyframe once the car has moved or turned far enough.
  #afterStep(lidar) {
    if (!this.localised) return;
    this.local.integrate(lidar, this.sensorPose());
    if (this.lastKf < 0) { this.#addKeyframe(lidar); return; }
    const z = relPose(this.lastKfFront, this.pose());
    if (Math.hypot(z[0], z[1]) >= KF_DIST || Math.abs(z[2]) >= KF_TURN) this.#addKeyframe(lidar);
  }

  // Estimated sensor pose, for building the map.
  sensorPose() {
    const [x, y, th] = this.ekf.x, [ox, oy] = this.lidarOffset;
    const c = Math.cos(th), s = Math.sin(th);
    return { x: x + c * ox - s * oy, y: y + s * ox + c * oy, yaw: th };
  }

  pose() { return this.ekf.x.slice(0, 3); }
}
