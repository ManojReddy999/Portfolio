// Loop closure: recognise a place mapped long ago and say exactly where the car
// is relative to it.
//
// For the newest keyframe, the nearest old keyframe (far back along the route,
// so outside the front end's local map) is a candidate if the graph puts it
// within reach of the accumulated drift. A small map is built from the old
// keyframe and its neighbours, and the new keyframe's scan is matched against it
// over metres and tens of degrees. A wrong closure would bend the whole map, so
// one match is never enough: a second keyframe a little further on must imply
// the same correction before either is added to the graph.

import { OccupancyGrid } from './mapping.js';
import { ScanMatcher } from './matcher.js';
import { relPose, compose } from './graph.js';

const MIN_GAP = 60; // keyframes (~24 m of route) between a keyframe and its loop candidates
const SUBMAP_SPAN = 12; // keyframes either side of the candidate in its submap
const MIN_SCORE = 0.3; // mean likelihood per return (a tracking match scores ~0.3 when aligned)
const MIN_POINTS = 80;
const MIN_OCCUPIED = 200; // cells in the candidate submap
const AGREE = [0.08, 0.05]; // m, rad: two matches must imply the same correction to within this
const PAIR_WITHIN = 8; // keyframes: how close the confirming match must follow
export const LOOP_SIGMA = [0.03, 0.03, 0.01]; // m, m, rad

export class LoopCloser {
  constructor(graph, lidarOffset, grid) {
    this.graph = graph;
    this.lidarOffset = lidarOffset;
    this.submap = new OccupancyGrid({ half: grid.half, res: grid.res });
    this.matcher = new ScanMatcher(this.submap, lidarOffset);
    this.pending = null;
    this.closures = 0; // accepted pairs
    this.unconfirmed = 0; // single matches that no second match agreed with
    this.searches = 0;
  }

  reset() { this.pending = null; }

  sensorPose(pose) {
    const [ox, oy] = this.lidarOffset, c = Math.cos(pose[2]), s = Math.sin(pose[2]);
    return { x: pose[0] + c * ox - s * oy, y: pose[1] + s * ox + c * oy, yaw: pose[2] };
  }

  // Look for a closure from keyframe k. `radius` is how far the drift may have
  // carried the estimate since the route was last pinned down.
  // Returns the constraints to add (two of them once confirmed), or null.
  attempt(k, radius) {
    const nodes = this.graph.nodes, me = nodes[k];
    let best = -1, bestD = Infinity;
    for (let c = 0; c < nodes.length; c++) {
      const nd = nodes[c];
      if (nd.segment === me.segment && k - c < MIN_GAP) continue;
      const d = Math.hypot(nd.pose[0] - me.pose[0], nd.pose[1] - me.pose[1]);
      if (d < radius && d < bestD) { best = c; bestD = d; }
    }
    if (best < 0) return null;
    this.searches++;

    // The candidate's neighbourhood, at its current graph poses.
    this.submap.clear();
    for (let c = Math.max(0, best - SUBMAP_SPAN); c <= Math.min(nodes.length - 1, best + SUBMAP_SPAN); c++) {
      const nd = nodes[c];
      if (nd.segment !== nodes[best].segment || (nd.segment === me.segment && k - c < MIN_GAP)) continue;
      this.submap.integrate(nd.scan, this.sensorPose(nd.pose));
    }
    this.matcher.refreshField(true);
    if (this.matcher.occupied < MIN_OCCUPIED) return null;

    const m = this.matcher.matchWide(me.scan, me.pose, [Math.min(radius, 2.0), 0.35]);
    if (m.score < MIN_SCORE || m.points < MIN_POINTS) return null;

    const found = [m.x, m.y, m.th];
    const z = relPose(nodes[best].pose, found); // the constraint: k as seen from the old keyframe
    const hit = { k, c: best, z, pose: me.pose.slice(), found, score: m.score };

    // Agreement: the rigid correction the first match implies, applied to this
    // keyframe's estimate, must land where this match found it.
    const p = this.pending;
    const predicted = p && compose(p.found, relPose(p.pose, me.pose));
    if (p && k - p.k <= PAIR_WITHIN && Math.abs(p.c - best) <= 2 * SUBMAP_SPAN
      && Math.hypot(predicted[0] - found[0], predicted[1] - found[1]) < AGREE[0]
      && Math.abs(Math.atan2(Math.sin(predicted[2] - found[2]), Math.cos(predicted[2] - found[2]))) < AGREE[1]) {
      this.pending = null;
      this.closures++;
      return [p, hit];
    }
    if (p && k - p.k > PAIR_WITHIN) this.unconfirmed++;
    this.pending = hit;
    return null;
  }
}
