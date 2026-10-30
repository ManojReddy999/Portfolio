// MuJoCo simulation of the training scene, running in the browser.
//
// The scene is exported from the training code by tools/export_scene.py, so the
// physics here is the physics the navigation policy was trained in.

import loadMujoco from '../vendor/mujoco/mujoco.js';

export const PHYSICS_STEPS_PER_CONTROL = 10; // 0.05 s control over 0.005 s physics
export const STEER_LIMIT = 0.38;
export const THROTTLE_RANGE = [-1, 3];

// Resolved against this module, so the lab can be embedded from any page.
const SCENE = new URL('../assets/scene/', import.meta.url);

// Meshes ship gzipped (GitHub Pages doesn't compress STL); inflate them here.
async function fetchBytes(name) {
  const res = await fetch(new URL(name, SCENE));
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  const body = name.endsWith('.gz') ? res.body.pipeThrough(new DecompressionStream('gzip')) : res.body;
  return new Uint8Array(await new Response(body).arrayBuffer());
}

export async function loadSim() {
  const [mujoco, manifest] = await Promise.all([
    loadMujoco(),
    fetch(new URL('manifest.json', SCENE)).then((r) => r.json()),
  ]);
  const files = await Promise.all(manifest.files.map(fetchBytes));

  const vfs = new mujoco.MjVFS();
  let xml = null;
  manifest.files.forEach((name, i) => {
    if (name === manifest.scene) xml = new TextDecoder().decode(files[i]);
    else vfs.addBuffer(name.replace(/\.gz$/, ''), files[i]);
  });
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  vfs.delete();
  return new Sim(mujoco, model, manifest);
}

export class Sim {
  constructor(mujoco, model, manifest) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.manifest = manifest;

    const obj = mujoco.mjtObj;
    this.carBody = mujoco.mj_name2id(model, obj.mjOBJ_BODY.value, 'car/buddy');
    this.lidarGeom = mujoco.mj_name2id(model, obj.mjOBJ_GEOM.value, 'car/buddy_ydlidar');
    if (this.carBody < 0 || this.lidarGeom < 0) throw new Error('car or lidar not found in scene');

    // Every body in the car's subtree, so contacts and rays can tell car from world.
    const parent = model.body_parentid;
    this.carBodies = new Set();
    for (let b = 0; b < model.nbody; b++) {
      for (let p = b; p > 0; p = parent[p]) {
        if (p === this.carBody) { this.carBodies.add(b); break; }
      }
    }
    // dm_control drops the car in with a single free joint; find where its pose lives.
    const free = mujoco.mjtJoint.mjJNT_FREE.value;
    this.freeJoint = -1;
    for (let j = 0; j < model.njnt; j++) if (model.jnt_type[j] === free) { this.freeJoint = j; break; }
    if (this.freeJoint < 0) throw new Error('car free joint not found');
    this.qposAdr = model.jnt_qposadr[this.freeJoint];

    // Obstacles: static geoms off the car that are not the floor or the arena walls.
    const plane = mujoco.mjtGeom.mjGEOM_PLANE.value;
    this.obstacles = [];
    for (let g = 0; g < model.ngeom; g++) {
      if (this.carBodies.has(model.geom_bodyid[g]) || model.geom_type[g] === plane) continue;
      const s = model.geom_size;
      if (Math.max(s[3 * g], s[3 * g + 1], s[3 * g + 2]) < 1) this.obstacles.push(g);
    }
    this.reset();
    this.restZ = this.data.qpos[this.qposAdr + 2];
  }

  // Re-scatter obstacles with the training rule (task.py): uniform over the
  // arena, at least 0.5 m from the start, centres at z = 0.2.
  scatterObstacles(rng, arenaSize) {
    const m = this.model, xpos = this.data.xpos, gpos = m.geom_pos;
    for (const g of this.obstacles) {
      let x, y;
      do {
        x = (rng() * 2 - 1) * arenaSize;
        y = (rng() * 2 - 1) * arenaSize;
      } while (Math.hypot(x, y) < 0.5);
      const b = 3 * m.geom_bodyid[g];
      gpos[3 * g] = x - xpos[b];
      gpos[3 * g + 1] = y - xpos[b + 1];
      gpos[3 * g + 2] = 0.2 - xpos[b + 2];
    }
    this.mujoco.mj_forward(m, this.data);
  }

  // Put the car at (x, y) facing yaw, at rest.
  placeCar(x, y, yaw) {
    const q = this.data.qpos, a = this.qposAdr;
    q[a] = x; q[a + 1] = y; q[a + 2] = this.restZ;
    q[a + 3] = Math.cos(yaw / 2); q[a + 4] = 0; q[a + 5] = 0; q[a + 6] = Math.sin(yaw / 2);
    this.data.qvel.fill(0);
    this.mujoco.mj_forward(this.model, this.data);
  }

  // Cosine of the car's tilt from upright; well below 1 means it has rolled or flipped.
  carUpright() {
    const [w, x, y] = this.carQuat();
    return 1 - 2 * (x * x + y * y);
  }

  reset() {
    this.mujoco.mj_resetData(this.model, this.data);
    this.mujoco.mj_forward(this.model, this.data);
  }

  // One 20 Hz control step: hold the action across the physics substeps.
  // `onSubstep(dt)` runs after each physics step, e.g. to integrate odometry at 200 Hz.
  step(steer, throttle, onSubstep = null) {
    const ctrl = this.data.ctrl, dt = this.model.opt.timestep;
    ctrl[0] = Math.max(-STEER_LIMIT, Math.min(STEER_LIMIT, steer));
    ctrl[1] = Math.max(THROTTLE_RANGE[0], Math.min(THROTTLE_RANGE[1], throttle));
    for (let i = 0; i < PHYSICS_STEPS_PER_CONTROL; i++) {
      this.mujoco.mj_step(this.model, this.data);
      if (onSubstep) onSubstep(dt);
    }
  }

  // Where the LiDAR sits in the car's body frame, measured from the model
  // (MuJoCo recentres the YDLidar mesh, so it is not the XML position).
  lidarOffset() {
    const g = this.data.geom_xpos, i = 3 * this.lidarGeom;
    const [bx, by, yaw] = this.carPose();
    const dx = g[i] - bx, dy = g[i + 1] - by, c = Math.cos(yaw), s = Math.sin(yaw);
    return [c * dx + s * dy, -s * dx + c * dy];
  }

  // True planar pose of the car body: [x, y, yaw].
  carPose() {
    const [x, y] = this.carPosition();
    return [x, y, this.carYaw()];
  }

  get time() { return this.data.time; }

  carPosition() {
    const p = this.data.xpos, i = 3 * this.carBody;
    return [p[i], p[i + 1], p[i + 2]];
  }

  carQuat() {
    const q = this.data.xquat, i = 4 * this.carBody;
    return [q[i], q[i + 1], q[i + 2], q[i + 3]];
  }

  carYaw() {
    const [w, x, y, z] = this.carQuat();
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  // Forward speed along the car's heading, from the simulator's free-joint velocity.
  carForwardSpeed() {
    const v = this.data.qvel, yaw = this.carYaw();
    return v[0] * Math.cos(yaw) + v[1] * Math.sin(yaw);
  }

  // True if any car geom is touching a non-car geom other than the floor.
  inCollision() {
    const m = this.model, contacts = this.data.contact;
    try {
      for (let i = 0, n = contacts.size(); i < n; i++) {
        const c = contacts.get(i);
        const a = m.geom_bodyid[c.geom1], b = m.geom_bodyid[c.geom2];
        const aCar = this.carBodies.has(a), bCar = this.carBodies.has(b);
        if (aCar === bCar) continue;
        const other = aCar ? c.geom2 : c.geom1;
        if (m.geom_type[other] !== this.mujoco.mjtGeom.mjGEOM_PLANE.value) return true;
      }
      return false;
    } finally {
      contacts.delete();
    }
  }
}
