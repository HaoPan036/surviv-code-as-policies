"""
Rule-based coded agent for Surviv RL.

Behaviour priority (highest first):
  1. Flee poison  — turn toward map centre and walk in when near the edge
  2. Dodge bullet — strafe away from any close incoming bullet
  3. Engage enemy — turn to aim; fire when centred; advance
  4. Idle scan    — spin a full 360° while randomly wandering
"""

import math
import random

from game import FOV_HALF_DEG, FOV_DEG_STEP

# ---------------------------------------------------------------------------
# Ray layout: 31 rays, -30° … 0° … +30° relative to facing direction.
# Index 0  = far left (-30°)
# Index 15 = dead ahead (0°)
# Index 30 = far right (+30°)
# ---------------------------------------------------------------------------
N_RAYS  = 2 * FOV_HALF_DEG // FOV_DEG_STEP + 1   # 31
CENTER  = N_RAYS // 2                               # 15

AIM_TOL       = 2     # ±rays around centre that count as "aimed" (±4°)
DODGE_DIST    = 0.35  # normalised FOV dist — bullet closer than this triggers dodge
POISON_MARGIN = 80    # units — start fleeing when this close to zone edge
ENGAGE_DIST   = 0.6   # normalised target distance to maintain on enemy (±0.08 dead zone)
ENGAGE_TOL    = 0.08  # normalised dist band around ENGAGE_DIST with no fwd/back


def _angle_diff(target: float, current: float) -> float:
    """Signed shortest angular distance (target − current) in (−π, π]."""
    d = (target - current) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


class CodedPolicy:
    def __init__(self):
        # Idle scan: spin one full revolution then flip direction
        self._scan_dir   =  1        # +1 → turn right (e), −1 → turn left (q)
        self._scan_steps =  0        # steps since last direction flip
        # ~180° per half-cycle at 150°/s × (1/60)s = 2.5°/step → 72 steps/half-turn
        self._scan_half  = 72

        # Random wander state
        self._wander_w   = True
        self._wander_side_key = 'a'
        self._wander_side = False
        self._wander_ttl  = 0

    def step(self, obs_raw: dict | None, player, game) -> dict:
        keys = {k: False for k in ['w', 's', 'a', 'd', 'q', 'e', 'space']}
        if obs_raw is None:
            return keys

        rays = obs_raw['rays']

        enemy_rays  = [(i, r) for i, r in enumerate(rays) if r['type'] == 'enemy']
        bullet_rays = [(i, r) for i, r in enumerate(rays) if r['type'] == 'bullet']

        # ── 1. Poison avoidance ──────────────────────────────────────────────
        po = game.poison
        to_cx = po.cx - player.x
        to_cy = po.cy - player.y
        if math.hypot(to_cx, to_cy) + POISON_MARGIN > po.radius:
            diff = _angle_diff(math.atan2(to_cy, to_cx), player.angle)
            if diff > 0.15:
                keys['e'] = True
            elif diff < -0.15:
                keys['q'] = True
            keys['w'] = True
            return keys

        # ── 2. Bullet dodge ──────────────────────────────────────────────────
        close = [(i, r) for i, r in bullet_rays if r['dist'] < DODGE_DIST]
        if close:
            i_b, _ = min(close, key=lambda x: x[1]['dist'])
            # Bullet on left side → strafe right, and vice-versa
            keys['a' if i_b >= CENTER else 'd'] = True
            keys['w'] = True
            return keys

        # ── 3. Enemy engagement ──────────────────────────────────────────────
        if enemy_rays:
            i_e, r_e = min(enemy_rays, key=lambda x: x[1]['dist'])
            offset = i_e - CENTER   # negative = enemy is to our left
            if abs(offset) <= AIM_TOL:
                keys['space'] = True
            elif offset < 0:
                keys['q'] = True    # turn left to track
            else:
                keys['e'] = True    # turn right to track
            # Hold ~60% of FOV range: advance if too far, back off if too close
            dist_err = r_e['dist'] - ENGAGE_DIST
            if dist_err > ENGAGE_TOL:
                keys['w'] = True    # too far — close in
            elif dist_err < -ENGAGE_TOL:
                keys['s'] = True    # too close — back off
            # Strafe to make a harder target
            if self._wander_ttl <= 0:
                self._wander_side_key = random.choice(['a', 'd'])
                self._wander_ttl      = random.randint(20, 60)
            self._wander_ttl -= 1
            keys[self._wander_side_key] = True
            return keys

        # ── 4. Idle: 360° scan + random wander ──────────────────────────────
        # Spin: flip direction every half-revolution so we sweep ±180°
        self._scan_steps += 1
        if self._scan_steps >= self._scan_half:
            self._scan_dir   = -self._scan_dir
            self._scan_steps = 0
        keys['e' if self._scan_dir > 0 else 'q'] = True

        # Random walk: re-roll every 20–80 steps
        if self._wander_ttl <= 0:
            self._wander_w        = random.random() < 0.7
            self._wander_side     = random.random() < 0.3
            self._wander_side_key = random.choice(['a', 'd'])
            self._wander_ttl      = random.randint(20, 80)
        self._wander_ttl -= 1

        if self._wander_w:
            keys['w'] = True
        if self._wander_side:
            keys[self._wander_side_key] = True

        return keys
