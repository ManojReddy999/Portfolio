// Log-odds occupancy grid built from LiDAR scans.
//
// Scans are integrated at whatever sensor pose they are given. With SLAM that
// is the car's ESTIMATED pose, so localisation error shows up in the map.

const L_FREE = -0.4;
const L_OCC = 0.9;
const L_MIN = -3;
const L_MAX = 5;

export class OccupancyGrid {
  constructor({ half = 8.6, res = 0.05 } = {}) {
    this.half = half;
    this.res = res;
    this.size = Math.ceil((2 * half) / res);
    this.logodds = new Float32Array(this.size * this.size);
    this.seen = new Uint8Array(this.size * this.size);
    this.seenCount = 0;
    this.version = 0; // bumps on every update
    this.epoch = 0; // bumps when the map is cleared
  }

  cell(x, y) {
    return [Math.floor((x + this.half) / this.res), Math.floor((y + this.half) / this.res)];
  }

  inside(i, j) {
    return i >= 0 && j >= 0 && i < this.size && j < this.size;
  }

  update(idx, delta) {
    const v = this.logodds[idx] + delta;
    this.logodds[idx] = v < L_MIN ? L_MIN : v > L_MAX ? L_MAX : v;
    if (!this.seen[idx]) { this.seen[idx] = 1; this.seenCount++; }
  }

  // Bresenham from the sensor to each return: free along the beam, occupied at a hit.
  // `pose` ({x, y, yaw} of the sensor) defaults to where the scan was really taken.
  integrate(lidar, pose = null) {
    const ox = pose ? pose.x : lidar.origin[0], oy = pose ? pose.y : lidar.origin[1];
    const yaw = pose ? pose.yaw : lidar.yaw;
    const [i0, j0] = this.cell(ox, oy);
    for (let k = 0; k < lidar.n; k++) {
      const a = yaw + (k * 2 * Math.PI) / lidar.n, r = lidar.ranges[k];
      const [i1, j1] = this.cell(ox + r * Math.cos(a), oy + r * Math.sin(a));
      let i = i0, j = j0;
      const di = Math.abs(i1 - i0), dj = Math.abs(j1 - j0);
      const si = i0 < i1 ? 1 : -1, sj = j0 < j1 ? 1 : -1;
      let err = di - dj;
      while (i !== i1 || j !== j1) {
        if (this.inside(i, j)) this.update(j * this.size + i, L_FREE);
        const e2 = 2 * err;
        if (e2 > -dj) { err -= dj; i += si; }
        if (e2 < di) { err += di; j += sj; }
      }
      if (this.inside(i1, j1)) this.update(j1 * this.size + i1, lidar.hit[k] ? L_OCC : L_FREE);
    }
    this.version++;
  }

  clear() {
    this.logodds.fill(0);
    this.seen.fill(0);
    this.seenCount = 0;
    this.version++;
    this.epoch++;
  }

  // Fraction of the walled arena that has been observed.
  coverage(arenaHalf) {
    const cellsInArena = Math.pow((2 * arenaHalf) / this.res, 2);
    return Math.min(1, this.seenCount / cellsInArena);
  }

  // Paint into an RGBA ImageData: occupied glows, observed-free is a faint tint, unknown is clear.
  paint(image, colors) {
    const { size, logodds, seen } = this;
    const px = image.data;
    for (let j = 0; j < size; j++) {
      const row = size - 1 - j; // canvas rows run top-down; +y is up
      for (let i = 0; i < size; i++) {
        const idx = j * size + i, o = 4 * (row * size + i);
        if (!seen[idx]) { px[o + 3] = 0; continue; }
        const l = logodds[idx];
        const c = l > 1.2 ? colors.occupied : l < -0.6 ? colors.free : colors.uncertain;
        px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = c[3];
      }
    }
  }
}
