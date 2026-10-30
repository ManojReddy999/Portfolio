// Stand-in driver: a classical arc-sampling local planner on the LiDAR scan.
//
// For each candidate steering angle it rolls the car's bicycle kinematics
// forward and checks the car's real footprint against every LiDAR return along
// the arc, then takes the longest collision-free arc toward open space. An
// Ackermann car cannot turn in place, so a planner has to reason about where
// its corners will sweep; a pure "steer toward the gap" rule clips obstacles
// with the front wheels.
//
// This is a hand-written placeholder so the scene can be judged before the
// trained policy is wired in. The trained policy reads the depth camera and
// replaces this module.

// Geometry of the MuSHR car (body frame, metres), from buddy.xml.
const LIDAR_X = -0.035; // sensor position along the body
const REAR_AXLE_X = -0.158;
const WHEELBASE = 0.1385 - REAR_AXLE_X;
const MARGIN = 0.03;
const FOOT = { xMin: -0.21 - MARGIN, xMax: 0.225 + MARGIN, halfW: 0.165 + MARGIN };

const STEER_LIMIT = 0.38;
const CANDIDATES = 15;
const HORIZON = 1.4; // metres of arc to check ahead
const DS = 0.04;
const REVERSE_HORIZON = 0.6;

export class LocalPlanner {
  constructor() {
    this.prevSteer = 0;
    this.reverseUntil = 0;
    this.reverseSteer = 0;
    this.stillSince = null;
    this.points = [];
  }

  // LiDAR returns in the car's body frame, nearby ones only.
  collectPoints(lidar) {
    const pts = [];
    for (let k = 0; k < lidar.n; k++) {
      if (!lidar.hit[k] || lidar.ranges[k] > HORIZON + 0.6) continue;
      const a = (k * 2 * Math.PI) / lidar.n, r = lidar.ranges[k];
      pts.push(LIDAR_X + r * Math.cos(a), r * Math.sin(a));
    }
    this.points = pts;
  }

  // Body-frame pose after travelling s along the rear-axle arc with steering delta.
  pose(delta, s) {
    const kappa = Math.tan(delta) / WHEELBASE;
    let rx, ry;
    const psi = kappa * s;
    if (Math.abs(kappa) < 1e-6) { rx = s; ry = 0; }
    else { rx = Math.sin(psi) / kappa; ry = (1 - Math.cos(psi)) / kappa; }
    // rear axle starts at (REAR_AXLE_X, 0); body origin sits -REAR_AXLE_X ahead of it
    const c = Math.cos(psi), sn = Math.sin(psi);
    return [REAR_AXLE_X + rx - REAR_AXLE_X * c, ry - REAR_AXLE_X * sn, psi];
  }

  // Does the car's footprint at body-frame pose (x, y, psi) contain any LiDAR return?
  blocked(x, y, psi) {
    const c = Math.cos(psi), s = Math.sin(psi), p = this.points;
    for (let i = 0; i < p.length; i += 2) {
      const dx = p[i] - x, dy = p[i + 1] - y;
      const u = c * dx + s * dy, v = -s * dx + c * dy;
      if (u > FOOT.xMin && u < FOOT.xMax && v > -FOOT.halfW && v < FOOT.halfW) return true;
    }
    return false;
  }

  // Collision-free arc length for a steering angle, driving forward (dir 1) or back (dir -1).
  freeLength(delta, dir, horizon) {
    for (let s = DS; s <= horizon; s += DS) {
      const [x, y, psi] = this.pose(delta, dir * s);
      if (this.blocked(x, y, psi)) return s - DS;
    }
    return horizon;
  }

  // `home`, if given, is { angle, weight }: the direction back toward the middle of the
  // arena in the body frame, and how strongly to prefer arcs that end facing it.
  act(lidar, t, speed, collided, home = null) {
    this.collectPoints(lidar);
    const steers = Array.from({ length: CANDIDATES }, (_, i) => -STEER_LIMIT + (2 * STEER_LIMIT * i) / (CANDIDATES - 1));

    // Long-range openness along each arc's end direction, so it heads toward space, not just away from walls.
    const openness = (delta) => {
      const [x, y] = this.pose(delta, HORIZON);
      let a = Math.atan2(y, x - LIDAR_X);
      if (a < 0) a += 2 * Math.PI;
      const k = Math.round((a / (2 * Math.PI)) * lidar.n) % lidar.n;
      return lidar.ranges[k] / lidar.range;
    };

    let best = 0, bestScore = -Infinity, bestFree = 0;
    for (const d of steers) {
      const free = this.freeLength(d, 1, HORIZON);
      let score = free / HORIZON + 0.35 * openness(d) - 0.08 * Math.abs(d) / STEER_LIMIT
        - 0.12 * Math.abs(d - this.prevSteer) / STEER_LIMIT;
      if (home) score += home.weight * Math.cos(this.pose(d, HORIZON)[2] - home.angle);
      if (score > bestScore) { bestScore = score; best = d; bestFree = free; }
    }

    if (Math.abs(speed) < 0.04) this.stillSince ??= t; else this.stillSince = null;
    const stuck = this.stillSince !== null && t - this.stillSince > 1.0;

    // Boxed in, touching something, or stuck: back out along the clearest reverse arc,
    // turning the nose toward the more open side.
    if (t >= this.reverseUntil && (bestFree < 0.18 || collided || stuck)) {
      let rBest = 0, rScore = -Infinity;
      for (const d of steers) {
        const free = this.freeLength(d, -1, REVERSE_HORIZON);
        // reversing with the wheel at -d swings the nose toward +d's side
        const score = free + 0.15 * openness(-d);
        if (score > rScore) { rScore = score; rBest = d; }
      }
      if (rScore > 0.08) {
        this.reverseUntil = t + 0.8;
        this.reverseSteer = rBest;
        this.stillSince = null;
      }
    }
    if (t < this.reverseUntil) {
      if (this.freeLength(this.reverseSteer, -1, 0.12) < 0.08) this.reverseUntil = t; // nothing more behind
      else return { steer: this.reverseSteer, throttle: -1, mode: 'reversing' };
    }

    this.prevSteer = best;
    const throttle = Math.max(0.45, Math.min(2.0, 0.45 + 1.35 * (bestFree - 0.25)));
    return { steer: best, throttle, mode: bestFree < 0.7 ? 'avoiding' : 'cruising' };
  }
}
