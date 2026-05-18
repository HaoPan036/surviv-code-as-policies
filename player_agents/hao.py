"""
hao.py — Code-as-Policies agent for Surviv RL.

Hand-written, no training. The policy is a small program composed of
named subroutines, selected by priority. The "LLM-as-programmer" writes
the controller; the game is the body.

Observation (per call to `step`):
  obs_raw['rays'] = 31 ray dicts, indexed 0..30
      idx  0 = far-left  (-30°)
      idx 15 = dead-ahead (AIM_C)
      idx 30 = far-right (+30°)
      each = {'type': 'empty'|'wall'|'poison'|'enemy'|'bullet',
              'dist':  normalised 0..1   (1.0 == 288 units, the FOV reach)}

Action keys returned:
  w / s        forward / back
  a / d        strafe left / right
  q / e        turn  left / right       (~2.5° per frame at 60 Hz)
  space        fire   (rate-limited by the engine at FIRE_CD = 0.35 s)

Behaviour priority (high → low):
  P1   RUN_HOME      — outside zone or near edge  → run toward centre
  P2   DODGE_BULLET  — incoming bullet close      → strafe perpendicular
  P3   ENGAGE        — enemy visible              → aim with lead, fire, kite
       └─ EXEC mode  — enemy visible ≥0.5 s AND no incoming bullet for ≥0.5 s
                       (i.e. they don't see us)  → sprint, no kite, dump bullets
  P4   PATROL        — nothing in cone            → sweep, drift toward centre

Multi-enemy support: when more than one enemy ray is visible, `_select_target`
picks the highest-threat one (close + near-centre), not just the nearest.
"""

import math
import random

from game import FOV_HALF_DEG, FOV_DEG_STEP, MAP_W, MAP_H

N_RAYS = 2 * FOV_HALF_DEG // FOV_DEG_STEP + 1   # 31
AIM_C  = N_RAYS // 2                             # 15  (dead ahead)

# Tunables -------------------------------------------------------------------
ENGAGE_DIST    = 0.22   # normalised distance we try to hold on an enemy
ENGAGE_BAND    = 0.10   # dead-zone around ENGAGE_DIST (no fwd/back)
DODGE_DIST     = 0.35   # bullets closer than this trigger an emergency dodge
POISON_MARGIN  = 50     # start running home this many units before the edge
LEAD_CAP_RAYS  = 4      # max bearing lead we will apply (≈ ±8°)
LEAD_MIN_DIST  = 0.40   # only lead at this distance or farther
LEAD_WARMUP    = 4      # frames of stable contact before trusting lead
PATROL_SWEEP   = 72     # ~half a revolution at 150°/s × (1/60)s
WALL_AHEAD_D   = 0.15   # forward wall closer than this ⇒ don't push 'w'
CLOSE_DIST     = 0.30   # below this an enemy counts as "in our face"
EXEC_CONTACT_T = 5      # frames of continuous enemy visibility before EXEC arms
EXEC_QUIET_T   = 25     # frames without any incoming bullet before EXEC fires
                        #   (NN FIRE_CD is 21 frames — keep > that to avoid
                        #    false triggers during their reload window)
FIRE_TOL_EXTRA = 2      # widen firing tolerance by this much beyond aim tol

# Battle-royale late-game trigger: when the safe-zone gets this small the
# field is reduced to a few survivors — switch from "selective" to "fight
# everyone" because there is no more time to wait others out.
PHASE_RUSH_R   = 200

# Distance-adaptive aim tolerance (rays around centre that count as on-target):
#   far enemies → tight aim (1 ray ≈ ±2°)
#   mid         → standard (2 rays ≈ ±4°)
#   close       → generous (3 rays ≈ ±6°) — almost any shot lands at point-blank
def _aim_tol(dist: float) -> int:
    if dist >= 0.70: return 1
    if dist >= 0.35: return 2
    return 3


def _wall_ahead(rays) -> bool:
    """True if a wall ray is close in the central ±2 indices."""
    for i in range(AIM_C - 2, AIM_C + 3):
        if rays[i]['type'] == 'wall' and rays[i]['dist'] < WALL_AHEAD_D:
            return True
    return False


def _count_enemy_clusters(enemies) -> int:
    """How many distinct enemies appear to be in our cone.

    Adjacent ray indices belong to the same body (a player at 18 u radius
    fills 2-4 rays). A gap > 3 between sorted enemy-ray indices means a
    different enemy.  Used to decide "fight or flee" in battle royale.
    """
    if not enemies:
        return 0
    idxs = sorted(i for i, _ in enemies)
    n = 1
    for k in range(1, len(idxs)):
        if idxs[k] - idxs[k - 1] > 3:
            n += 1
    return n


def _select_target(enemies):
    """Multi-enemy threat picker.

    Distance is the dominant factor; off-axis enemies get a small penalty
    because we'd have to swing the gun to reach them (and they are less
    likely to be lined up to shoot us right now).
        score(idx, r) = r['dist'] · (1 + 0.5·|idx − centre|/centre)
    Lower score = pick first.
    """
    def threat(item):
        idx, r = item
        bearing_pen = abs(idx - AIM_C) / AIM_C    # 0.0 centre … 1.0 edge
        return r['dist'] * (1.0 + 0.5 * bearing_pen)
    return min(enemies, key=threat)


def _angle_diff(target: float, current: float) -> float:
    """Signed shortest angular distance, in (-π, π]."""
    d = (target - current) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


def _empty_keys() -> dict:
    return {k: False for k in ('w', 's', 'a', 'd', 'q', 'e', 'space')}


def _turn_toward(keys: dict, signed_offset: float, dead_zone: float = 0.0) -> None:
    """signed_offset > 0 ⇒ target is to the right ⇒ press 'e'."""
    if signed_offset >  dead_zone:
        keys['e'] = True
    elif signed_offset < -dead_zone:
        keys['q'] = True


class HaoPolicy:
    """Hand-coded policy. Drop-in replacement for CodedPolicy."""

    def __init__(self):
        self._prev_enemy_idx: int | None = None   # for bearing-rate / leading
        self._bearing_rate: float        = 0.0    # smoothed rays-per-step
        self._contact_t  : int           = 0      # consecutive frames of stable enemy contact
        self._patrol_dir : int           = 1      # +1 = right (e), -1 = left (q)
        self._patrol_t   : int           = 0      # frames since last flip
        self._kite_side  : str           = 'a'    # 'a' or 'd', refreshed periodically
        self._kite_ttl   : int           = 0
        # EXEC-mode signals: an enemy in our cone who never shoots at us is
        # almost certainly facing away — we charge them head-on.
        self._enemy_seen_t: int  = 0              # consecutive frames any enemy visible
        self._no_bullet_t : int  = 999            # frames since any enemy bullet in cone

    # ------------------------------------------------------------------ main
    def step(self, obs_raw: dict | None, player, game) -> dict:
        keys = _empty_keys()
        if obs_raw is None:
            return keys

        rays      = obs_raw['rays']
        enemies   = [(i, r) for i, r in enumerate(rays) if r['type'] == 'enemy']
        bullets   = [(i, r) for i, r in enumerate(rays) if r['type'] == 'bullet']
        wall_fwd  = _wall_ahead(rays)

        # Update EXEC-mode signals every frame.
        self._enemy_seen_t = self._enemy_seen_t + 1 if enemies else 0
        self._no_bullet_t  = 0 if bullets else self._no_bullet_t + 1
        exec_mode = (self._enemy_seen_t >= EXEC_CONTACT_T
                     and self._no_bullet_t >= EXEC_QUIET_T)

        # Late-game escalation: when zone is small, the field is down to a
        # couple of agents. Multi-enemy retreat would just let them whittle
        # us down with poison — engage everything.
        phase_rush = game.poison.radius < PHASE_RUSH_R

        # P1 ── run home if we are about to be caught outside the safe zone -
        if self._run_home(keys, player, game, wall_fwd):
            self._forget_enemy()
            return keys

        # P2 ── dodge an incoming bullet ------------------------------------
        if self._dodge_bullet(keys, bullets):
            # Opportunistic shot while dodging: trigger only on the highest-
            # threat enemy if they happen to be in our crosshair.
            if enemies:
                i_e, r_e = _select_target(enemies)
                c = rays[AIM_C]
                clear = not (c['type'] == 'wall' and c['dist'] < r_e['dist'])
                if abs(i_e - AIM_C) <= _aim_tol(r_e['dist']) and clear:
                    keys['space'] = True
            return keys

        # P3 ── BR-aware engage:
        #   default       — 1 enemy → engage; ≥2 → retreat (let them trade)
        #   LATE override — radius < 200 → engage everyone (no time to camp)
        if enemies:
            multi = _count_enemy_clusters(enemies) >= 2
            if multi and not phase_rush:
                self._multi_retreat(keys, enemies, player, game)
                return keys
            self._engage(keys, enemies, rays, wall_fwd, exec_mode)
            return keys

        # P4 ── nothing in cone: sweep & drift toward map centre -----------
        self._forget_enemy()
        self._patrol(keys, player, wall_fwd)
        return keys

    # ------------------------------------------------------------ behaviours
    def _run_home(self, keys, player, game, wall_fwd: bool) -> bool:
        po   = game.poison
        d    = math.hypot(po.cx - player.x, po.cy - player.y)
        if d + POISON_MARGIN <= po.radius:
            return False                            # safe enough, ignore
        bearing = math.atan2(po.cy - player.y, po.cx - player.x)
        diff    = _angle_diff(bearing, player.angle)
        _turn_toward(keys, diff, dead_zone=0.10)
        if wall_fwd:
            # Wall blocks straight-line approach: side-step around it.
            keys['d' if diff >= 0 else 'a'] = True
        elif abs(diff) > math.radians(150):
            # Centre is behind us — walk backwards while spinning round.
            keys['s'] = True
        else:
            keys['w'] = True
        return True

    def _dodge_bullet(self, keys, bullets) -> bool:
        close = [(i, r) for i, r in bullets if r['dist'] < DODGE_DIST]
        if not close:
            return False
        i_b, _ = min(close, key=lambda x: x[1]['dist'])
        # Strafe perpendicular to the threat:
        #   bullet on the right half of the cone (i_b > AIM_C) → strafe left.
        keys['a' if i_b >= AIM_C else 'd'] = True
        # Twitch forward a touch so we don't sit still and eat the next shot.
        keys['w'] = True
        return True

    def _engage(self, keys, enemies, rays, wall_fwd: bool, exec_mode: bool) -> None:
        i_e, r_e = _select_target(enemies)
        dist     = r_e['dist']
        aim_tol  = _aim_tol(dist)
        c        = rays[AIM_C]
        blocked  = c['type'] == 'wall' and c['dist'] < dist
        raw_off  = i_e - AIM_C

        # ── EXEC mode ─────────────────────────────────────────────────────
        # The enemy is visible but hasn't fired at us — they don't see us.
        # No kite, no lead: sprint head-on and dump bullets at point-blank.
        if exec_mode:
            if not blocked:
                keys['space'] = True
            _turn_toward(keys, raw_off)
            if not wall_fwd:
                keys['w'] = True
            self._prev_enemy_idx = i_e          # keep bearing state warm
            return

        # ── Bearing-rate estimate (EMA) → lead the shot --------------------
        if self._prev_enemy_idx is not None:
            inst = i_e - self._prev_enemy_idx
            if abs(inst) <= 4:
                self._bearing_rate = 0.7 * self._bearing_rate + 0.3 * inst
                self._contact_t   += 1
            else:
                self._bearing_rate = 0.0
                self._contact_t    = 1
        else:
            self._contact_t = 1
        self._prev_enemy_idx = i_e

        if self._contact_t >= LEAD_WARMUP and dist >= LEAD_MIN_DIST:
            flight_frames = dist * 28.8              # 288u / 600u·s⁻¹ · 60 fps
            lead = int(round(self._bearing_rate * flight_frames))
            lead = max(-LEAD_CAP_RAYS, min(LEAD_CAP_RAYS, lead))
        else:
            lead = 0
        aim_idx = i_e + lead
        led_off = aim_idx - AIM_C

        # Sustained fire: widen the firing tolerance beyond strict aim tol so
        # we use every FIRE_CD window even if our aim is mid-correction.
        fire_tol = aim_tol + FIRE_TOL_EXTRA
        if (abs(raw_off) <= fire_tol or abs(led_off) <= fire_tol) and not blocked:
            keys['space'] = True

        # Aim damping at close range: in firing zone already → don't twitch.
        if not (dist < CLOSE_DIST and abs(raw_off) <= aim_tol):
            _turn_toward(keys, led_off)

        # Range control.
        derr = dist - ENGAGE_DIST
        if   derr >  ENGAGE_BAND and not wall_fwd:
            keys['w'] = True
        elif derr < -ENGAGE_BAND:
            keys['s'] = True

        # Kite with short fully-random strafe pulses → denies NN a clean
        # bearing-rate signal it could feed into its own lead estimator.
        if self._kite_ttl <= 0:
            self._kite_side = random.choice(('a', 'd'))
            self._kite_ttl  = random.randint(8, 18)
        self._kite_ttl -= 1
        keys[self._kite_side] = True

    def _multi_retreat(self, keys, enemies, player, game) -> None:
        """Battle-royale rule: when 2+ enemies are in our cone, do NOT fight.
        Let them shoot each other while we slip away. We strafe to the side
        opposite the group AND prefer moving toward the safe-zone centre so
        poison doesn't punish the retreat.
        """
        avg_idx = sum(i for i, _ in enemies) / len(enemies)
        # Strafe to the side away from the enemy group.
        keys['a' if avg_idx >= AIM_C else 'd'] = True
        # Back-pedal in the direction of map centre when possible — keeps us
        # inside the zone for the inevitable poison crunch.
        po = game.poison
        to_cx = po.cx - player.x
        to_cy = po.cy - player.y
        bearing = math.atan2(to_cy, to_cx)
        diff    = _angle_diff(bearing, player.angle)
        if abs(diff) > math.radians(120):
            # Centre is roughly behind → backpedal carries us inward.
            keys['s'] = True
        elif abs(diff) < math.radians(60):
            # Centre is in front → cautious forward (toward zone).
            keys['w'] = True
        else:
            # Centre is to one side → backpedal is still safer than forward.
            keys['s'] = True
        # No firing — we don't want to alert anyone or eat return-fire.
        self._forget_enemy()

    def _patrol(self, keys, player, wall_fwd: bool) -> None:
        # Sweep ±180° looking for someone.
        self._patrol_t += 1
        if self._patrol_t >= PATROL_SWEEP:
            self._patrol_dir = -self._patrol_dir
            self._patrol_t   = 0
        keys['e' if self._patrol_dir > 0 else 'q'] = True

        # While searching, drift toward the centre of the map so we are
        # already well placed when the zone closes.
        dx = MAP_W / 2 - player.x
        dy = MAP_H / 2 - player.y
        if dx * dx + dy * dy > 150 * 150:
            home_bearing = math.atan2(dy, dx)
            diff = _angle_diff(home_bearing, player.angle)
            # Walk forward when home is mostly ahead — but not into a wall.
            if abs(diff) < math.radians(60) and not wall_fwd:
                keys['w'] = True
            elif wall_fwd:
                # Slide along the wall in whichever direction is "homeward".
                keys['d' if diff >= 0 else 'a'] = True

    # -------------------------------------------------------------- helpers
    def _forget_enemy(self) -> None:
        self._prev_enemy_idx = None
        self._bearing_rate   = 0.0
        self._contact_t      = 0


# Convenience alias: harness loaders that look for a class named `Policy`
# will pick up `HaoPolicy` automatically.
Policy = HaoPolicy