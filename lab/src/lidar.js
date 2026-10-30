// Simulated 2D LiDAR mounted where the MuSHR car carries its YDLidar.
//
// Rays are cast by MuJoCo itself against the same geometry the physics uses,
// in the horizontal plane through the sensor.
//
// One mj_ray per beam rather than mj_multiRay: in MuJoCo 3.12, mj_multiRay
// misses static geoms after they are relocated (the arena is re-scattered per
// visit), while mj_ray agrees with the physics. 360 beams cost about 2 ms.

const VISIBLE_GROUPS = [1, 0, 1, 1, 1, 1]; // skip group 1: the car's visual-only wheel meshes

export class Lidar {
  constructor(sim, { rays = 360, range = 4.0 } = {}) {
    const mj = sim.mujoco;
    this.sim = sim;
    this.n = rays;
    this.range = range;
    this.geomid = new mj.IntBuffer(1);
    this.normal = new mj.DoubleBuffer(3);
    this.dir = [0, 0, 0];

    this.ranges = new Float32Array(rays); // metres, index 0 = straight ahead, counter-clockwise
    this.hit = new Uint8Array(rays); // 1 where the ray struck something within range
    this.origin = [0, 0, 0]; // world position of the sensor at the last scan
    this.yaw = 0; // car heading at the last scan
  }

  angle(k) {
    const a = (k * 2 * Math.PI) / this.n;
    return a > Math.PI ? a - 2 * Math.PI : a; // (-pi, pi]
  }

  scan() {
    const { sim, n, range, dir } = this;
    const mj = sim.mujoco, xp = sim.data.geom_xpos, g = 3 * sim.lidarGeom;
    this.origin = [xp[g], xp[g + 1], xp[g + 2]];
    this.yaw = sim.carYaw();

    for (let k = 0; k < n; k++) {
      const a = this.yaw + (k * 2 * Math.PI) / n;
      dir[0] = Math.cos(a);
      dir[1] = Math.sin(a);
      const d = mj.mj_ray(sim.model, sim.data, this.origin, dir, VISIBLE_GROUPS, true,
        sim.carBody, this.geomid, this.normal);
      const hit = d >= 0 && d < range;
      this.hit[k] = hit ? 1 : 0;
      this.ranges[k] = hit ? d : range;
    }
  }

  // World-frame end point of ray k from the last scan.
  endpoint(k) {
    const a = this.yaw + (k * 2 * Math.PI) / this.n;
    return [this.origin[0] + this.ranges[k] * Math.cos(a), this.origin[1] + this.ranges[k] * Math.sin(a)];
  }

  dispose() {
    this.geomid.delete();
    this.normal.delete();
  }
}
