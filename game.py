import math
import random
import numpy as np

MAP_W = 1000
MAP_H = 1000
PLAYER_R = 18
BULLET_R = 5
BULLET_SPEED = 600        # units / sec
PLAYER_SPEED = 200        # units / sec
TURN_RATE = math.radians(150)  # rad / sec
PLAYER_HP = 100
BULLET_DMG = 25           # 4 hits to kill
BULLET_LIFE = 2.5         # seconds before bullet expires
FIRE_CD = 0.35            # seconds between shots (~3/sec)
POISON_R0 = 720.0         # initial safe-zone radius (covers full map: diagonal from center ≈ 707)
POISON_GRACE = 20.0       # seconds before zone starts shrinking
POISON_T = 120.0          # seconds to fully close after grace period ends
POISON_DPS = 8            # HP / sec outside safe zone

# Observation / perception
FOV_RANGE    = 16 * PLAYER_R   # 288 units — must match frontend
FOV_DEG_STEP = 2               # degrees between rays
FOV_HALF_DEG = 30              # half-angle of cone
WALL_PEEK    = 10              # units to extend past wall surface (visual only)


# ---------------------------------------------------------------------------
# Ray-casting helpers
# ---------------------------------------------------------------------------

def _ray_vs_rect(ox, oy, dx, dy, obs):
    """Distance t where ray first hits a rotated rectangle (Inf if no hit)."""
    ca = math.cos(-obs.angle); sa = math.sin(-obs.angle)
    ex = ox - obs.cx;          ey = oy - obs.cy
    lox = ex*ca - ey*sa;       loy = ex*sa + ey*ca
    ldx = dx*ca - dy*sa;       ldy = dx*sa + dy*ca
    hw = obs.w / 2;            hh = obs.h / 2
    tmin = 0.0;                tmax = math.inf
    if abs(ldx) > 1e-10:
        t1 = (-hw - lox) / ldx; t2 = (hw - lox) / ldx
        tmin = max(tmin, min(t1, t2)); tmax = min(tmax, max(t1, t2))
    elif lox < -hw or lox > hw:
        return math.inf
    if abs(ldy) > 1e-10:
        t1 = (-hh - loy) / ldy; t2 = (hh - loy) / ldy
        tmin = max(tmin, min(t1, t2)); tmax = min(tmax, max(t1, t2))
    elif loy < -hh or loy > hh:
        return math.inf
    return tmin if tmax >= tmin else math.inf


def _ray_vs_circle(ox, oy, dx, dy, cx, cy, r):
    """Distance t where ray first hits a circle (Inf if no hit)."""
    ex = ox - cx; ey = oy - cy
    b = ex*dx + ey*dy
    c = ex*ex + ey*ey - r*r
    disc = b*b - c
    if disc < 0:
        return math.inf
    sq = math.sqrt(disc)
    t1, t2 = -b - sq, -b + sq
    if t1 >= 0: return t1
    if t2 >= 0: return t2
    return math.inf


def _ray_vs_map_boundary(ox, oy, dx, dy):
    """Distance t where ray exits the map rectangle."""
    t = math.inf
    if abs(dx) > 1e-10:
        tx = ((MAP_W if dx > 0 else 0) - ox) / dx
        if tx > 0: t = min(t, tx)
    if abs(dy) > 1e-10:
        ty = ((MAP_H if dy > 0 else 0) - oy) / dy
        if ty > 0: t = min(t, ty)
    return t


# ---------------------------------------------------------------------------
# Obstacle helpers (player / bullet collision)
# ---------------------------------------------------------------------------

def _circle_vs_rect(px, py, pr, obs):
    """
    Circle (px,py,pr) vs rotated rectangle obstacle.
    Returns (colliding, push_world_x, push_world_y).
    push_* is the vector to add to the circle center to resolve the overlap.
    """
    ca = math.cos(-obs.angle)
    sa = math.sin(-obs.angle)
    dx = px - obs.cx
    dy = py - obs.cy
    # Circle centre in rect local space
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca

    hw, hh = obs.w / 2, obs.h / 2
    clamped_x = max(-hw, min(hw, lx))
    clamped_y = max(-hh, min(hh, ly))

    ddx = lx - clamped_x
    ddy = ly - clamped_y
    dist_sq = ddx * ddx + ddy * ddy

    if dist_sq >= pr * pr:
        return False, 0.0, 0.0

    if dist_sq > 1e-9:
        dist = math.sqrt(dist_sq)
        lpx = ddx / dist * (pr - dist)
        lpy = ddy / dist * (pr - dist)
    else:
        # Centre is inside rect — push out along shortest overlap axis
        ox = hw - abs(lx) + pr
        oy = hh - abs(ly) + pr
        if ox < oy:
            lpx, lpy = math.copysign(ox, lx), 0.0
        else:
            lpx, lpy = 0.0, math.copysign(oy, ly)

    # Rotate push back to world space
    ca2 = math.cos(obs.angle)
    sa2 = math.sin(obs.angle)
    return True, lpx * ca2 - lpy * sa2, lpx * sa2 + lpy * ca2


def _point_in_rect(px, py, obs):
    ca = math.cos(-obs.angle)
    sa = math.sin(-obs.angle)
    dx = px - obs.cx
    dy = py - obs.cy
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    return abs(lx) <= obs.w / 2 and abs(ly) <= obs.h / 2


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class Obstacle:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w      # full width  (along local x)
        self.h = h      # full height (along local y)
        self.angle = angle  # radians

    def to_dict(self):
        return {'cx': self.cx, 'cy': self.cy,
                'w': self.w, 'h': self.h, 'angle': self.angle}


def _make_obstacles(n=18):
    """Generate n random obstacles scattered across the map."""
    obs = []
    for _ in range(n):
        cx = random.uniform(80, MAP_W - 80)
        cy = random.uniform(80, MAP_H - 80)
        # Pick a random "style": long thin wall, medium wall, or chunky block
        style = random.random()
        if style < 0.45:          # long thin wall
            w = random.uniform(160, 340)
            h = random.uniform(14, 26)
        elif style < 0.75:        # medium wall
            w = random.uniform(70, 170)
            h = random.uniform(22, 55)
        else:                     # chunky block
            w = random.uniform(45, 110)
            h = random.uniform(45, 110)
        angle = random.uniform(0, math.pi)
        obs.append(Obstacle(cx, cy, w, h, angle))
    return obs


class Player:
    def __init__(self, pid, x, y):
        self.id = pid
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.hp = float(PLAYER_HP)
        self.fire_cd = 0.0
        self.alive = True

    def to_dict(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'health': self.hp,
            'alive': self.alive,
        }


class Bullet:
    def __init__(self, bid, owner, x, y, angle):
        self.id = bid
        self.owner = owner
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * BULLET_SPEED
        self.vy = math.sin(angle) * BULLET_SPEED
        self.life = BULLET_LIFE

    def to_dict(self):
        return {'id': self.id, 'x': self.x, 'y': self.y}


class Poison:
    def __init__(self):
        self.cx = MAP_W / 2
        self.cy = MAP_H / 2
        self.radius = POISON_R0
        self._elapsed = 0.0

    def reset(self):
        self.radius = POISON_R0
        self._elapsed = 0.0

    def update(self, dt):
        self._elapsed += dt
        shrink_t = max(0.0, self._elapsed - POISON_GRACE)
        t = min(shrink_t / POISON_T, 1.0)
        self.radius = POISON_R0 * (1.0 - t)

    def to_dict(self):
        return {'cx': self.cx, 'cy': self.cy, 'radius': self.radius}


class Game:
    def __init__(self, headless: bool = False):
        self.headless = headless          # True → skip cone_pts (RL training mode)
        self.players: dict[str, Player] = {}
        self.bullets: dict[str, Bullet] = {}
        self.inputs: dict[str, dict] = {}
        self.poison = Poison()
        self.obstacles = _make_obstacles()
        self._bid = 0
        self._cache_obs_arrays()

    def _cache_obs_arrays(self):
        """Pre-build numpy arrays from obstacle geometry for vectorised raycasting."""
        obs = self.obstacles
        self._np_ca = np.cos([-o.angle for o in obs])
        self._np_sa = np.sin([-o.angle for o in obs])
        self._np_cx = np.array([o.cx   for o in obs])
        self._np_cy = np.array([o.cy   for o in obs])
        self._np_hw = np.array([o.w/2  for o in obs])
        self._np_hh = np.array([o.h/2  for o in obs])

    def reset(self):
        self.players.clear()
        self.inputs.clear()
        self.bullets.clear()
        self.poison.reset()
        self.obstacles = _make_obstacles()
        self._cache_obs_arrays()

    def add_player(self, pid, x=None, y=None):
        if x is None: x = random.uniform(120, MAP_W - 120)
        if y is None: y = random.uniform(120, MAP_H - 120)
        self.players[pid] = Player(pid, x, y)
        self.inputs[pid] = {}

    def remove_player(self, pid):
        self.players.pop(pid, None)
        self.inputs.pop(pid, None)

    def set_input(self, pid, keys: dict):
        self.inputs[pid] = keys

    def update(self, dt):
        self._move_players(dt)
        self._move_bullets(dt)
        self._collide_bullets()
        self.poison.update(dt)
        self._poison_damage(dt)

    # ------------------------------------------------------------------ #

    def _move_players(self, dt):
        for pid, p in self.players.items():
            if not p.alive:
                continue
            inp = self.inputs.get(pid, {})

            if inp.get('q'):
                p.angle -= TURN_RATE * dt
            if inp.get('e'):
                p.angle += TURN_RATE * dt

            ca, sa = math.cos(p.angle), math.sin(p.angle)
            dx = dy = 0.0
            if inp.get('w'):
                dx += ca; dy += sa
            if inp.get('s'):
                dx -= ca; dy -= sa
            if inp.get('a'):
                dx += sa; dy -= ca
            if inp.get('d'):
                dx -= sa; dy += ca

            mag = math.hypot(dx, dy)
            if mag:
                dx /= mag; dy /= mag

            p.x = max(PLAYER_R, min(MAP_W - PLAYER_R, p.x + dx * PLAYER_SPEED * dt))
            p.y = max(PLAYER_R, min(MAP_H - PLAYER_R, p.y + dy * PLAYER_SPEED * dt))

            # Obstacle collision resolution
            for obs in self.obstacles:
                hit, px, py = _circle_vs_rect(p.x, p.y, PLAYER_R, obs)
                if hit:
                    p.x += px
                    p.y += py
            p.x = max(PLAYER_R, min(MAP_W - PLAYER_R, p.x))
            p.y = max(PLAYER_R, min(MAP_H - PLAYER_R, p.y))

            p.fire_cd = max(0.0, p.fire_cd - dt)
            if inp.get('space') and p.fire_cd == 0.0:
                self._bid += 1
                self.bullets[str(self._bid)] = Bullet(str(self._bid), pid, p.x, p.y, p.angle)
                p.fire_cd = FIRE_CD

    def _move_bullets(self, dt):
        dead = []
        for bid, b in self.bullets.items():
            b.x += b.vx * dt
            b.y += b.vy * dt
            b.life -= dt
            if b.life <= 0 or not (0 <= b.x <= MAP_W) or not (0 <= b.y <= MAP_H):
                dead.append(bid)
                continue
            for obs in self.obstacles:
                if _point_in_rect(b.x, b.y, obs):
                    dead.append(bid)
                    break
        for bid in dead:
            self.bullets.pop(bid, None)

    def _collide_bullets(self):
        dead = []
        for bid, b in self.bullets.items():
            for pid, p in self.players.items():
                if pid == b.owner or not p.alive:
                    continue
                if math.hypot(b.x - p.x, b.y - p.y) < PLAYER_R + BULLET_R:
                    p.hp = max(0.0, p.hp - BULLET_DMG)
                    if p.hp == 0:
                        p.alive = False
                    dead.append(bid)
                    break
        for bid in dead:
            self.bullets.pop(bid, None)

    def _poison_damage(self, dt):
        po = self.poison
        for p in self.players.values():
            if not p.alive:
                continue
            if math.hypot(p.x - po.cx, p.y - po.cy) > po.radius:
                p.hp = max(0.0, p.hp - POISON_DPS * dt)
                if p.hp == 0:
                    p.alive = False

    def compute_observation(self, pid):
        """
        31-ray observation for player `pid` — fully vectorised over rays.
        Returns:
          rays      - [{type, dist}]  — RL observation, dist normalised 0-1
          cone_pts  - [[wx, wy], …]  — world-space polygon for vision mask
        """
        p = self.players.get(pid)
        if not p or not p.alive:
            return None

        N    = 2 * FOV_HALF_DEG // FOV_DEG_STEP + 1       # 31
        degs = np.arange(-FOV_HALF_DEG, FOV_HALF_DEG + 1, FOV_DEG_STEP)
        a    = p.angle + np.radians(degs)                  # (N,)
        dx   = np.cos(a)                                   # (N,)
        dy   = np.sin(a)

        best_t    = np.full(N, float(FOV_RANGE))
        best_type = np.zeros(N, dtype=np.int8)             # 0 = empty

        # ── map boundary ─────────────────────────────────────────────────────
        t = np.full(N, np.inf)
        mx = np.abs(dx) > 1e-10
        tx = np.where(dx > 0, MAP_W - p.x, -p.x) / np.where(mx, dx, 1.0)
        t  = np.where(mx & (tx > 0), np.minimum(t, tx), t)
        my = np.abs(dy) > 1e-10
        ty = np.where(dy > 0, MAP_H - p.y, -p.y) / np.where(my, dy, 1.0)
        t  = np.where(my & (ty > 0), np.minimum(t, ty), t)
        hit = t < best_t
        best_t    = np.where(hit, t, best_t)
        best_type = np.where(hit, 1, best_type)            # wall

        # ── all obstacles — (N rays) × (M obstacles) in one broadcast ────────
        if self.obstacles:
            # player-to-obstacle-centre in each obstacle's local frame → (M,)
            ox_  = p.x - self._np_cx
            oy_  = p.y - self._np_cy
            lox  = ox_ * self._np_ca - oy_ * self._np_sa   # (M,)
            loy  = ox_ * self._np_sa + oy_ * self._np_ca

            # ray directions in obstacle local frame → (N, M)
            ldx = dx[:, None] * self._np_ca - dy[:, None] * self._np_sa
            ldy = dx[:, None] * self._np_sa + dy[:, None] * self._np_ca

            # slab test
            tmin = np.zeros((N, len(self.obstacles)))
            tmax = np.full((N, len(self.obstacles)), np.inf)

            mk_x  = np.abs(ldx) > 1e-10
            sldx  = np.where(mk_x, ldx, 1.0)
            t1x   = (-self._np_hw - lox) / sldx            # (N, M)
            t2x   = ( self._np_hw - lox) / sldx
            tmin  = np.where(mk_x, np.maximum(tmin, np.minimum(t1x, t2x)), tmin)
            tmax  = np.where(mk_x, np.minimum(tmax, np.maximum(t1x, t2x)), tmax)
            tmax  = np.where(~mk_x & ((lox < -self._np_hw) | (lox > self._np_hw)),
                             -np.inf, tmax)

            mk_y  = np.abs(ldy) > 1e-10
            sldy  = np.where(mk_y, ldy, 1.0)
            t1y   = (-self._np_hh - loy) / sldy
            t2y   = ( self._np_hh - loy) / sldy
            tmin  = np.where(mk_y, np.maximum(tmin, np.minimum(t1y, t2y)), tmin)
            tmax  = np.where(mk_y, np.minimum(tmax, np.maximum(t1y, t2y)), tmax)
            tmax  = np.where(~mk_y & ((loy < -self._np_hh) | (loy > self._np_hh)),
                             -np.inf, tmax)

            t_obs = np.where((tmax >= tmin) & (tmin >= 0), tmin, np.inf)
            t     = t_obs.min(axis=1)                      # (N,) closest obstacle
            hit   = t < best_t
            best_t    = np.where(hit, t, best_t)
            best_type = np.where(hit, 1, best_type)

        # ── poison circle ────────────────────────────────────────────────────
        po  = self.poison
        ox_ = p.x - po.cx;  oy_ = p.y - po.cy
        qb  = ox_ * dx + oy_ * dy
        qc  = ox_*ox_ + oy_*oy_ - po.radius**2
        disc = qb*qb - qc
        sq   = np.sqrt(np.maximum(disc, 0.0))
        t    = np.where(disc < 0, np.inf,
               np.where(-qb - sq >= 0, -qb - sq,
               np.where(-qb + sq >= 0, -qb + sq, np.inf)))
        hit  = t < best_t
        best_t    = np.where(hit, t, best_t)
        best_type = np.where(hit, 2, best_type)            # poison

        # ── enemies ──────────────────────────────────────────────────────────
        enemies = [(o.x, o.y) for op, o in self.players.items()
                   if op != pid and o.alive]
        if enemies:
            ecx = np.array([e[0] for e in enemies])
            ecy = np.array([e[1] for e in enemies])
            ox_ = p.x - ecx;  oy_ = p.y - ecy             # (E,)
            qb  = ox_ * dx[:, None] + oy_ * dy[:, None]   # (N, E)
            qc  = ox_*ox_ + oy_*oy_ - PLAYER_R**2         # (E,)
            disc = qb*qb - qc
            sq   = np.sqrt(np.maximum(disc, 0.0))
            t1   = -qb - sq;  t2 = -qb + sq
            t_all = np.where(disc < 0, np.inf,
                    np.where(t1 >= 0, t1,
                    np.where(t2 >= 0, t2, np.inf)))
            t     = t_all.min(axis=1)
            hit   = t < best_t
            best_t    = np.where(hit, t, best_t)
            best_type = np.where(hit, 3, best_type)        # enemy

        # ── enemy bullets ────────────────────────────────────────────────────
        blts = [(bl.x, bl.y) for bl in self.bullets.values() if bl.owner != pid]
        if blts:
            bcx = np.array([b[0] for b in blts])
            bcy = np.array([b[1] for b in blts])
            ox_ = p.x - bcx;  oy_ = p.y - bcy
            qb  = ox_ * dx[:, None] + oy_ * dy[:, None]
            qc  = ox_*ox_ + oy_*oy_ - BULLET_R**2
            disc = qb*qb - qc
            sq   = np.sqrt(np.maximum(disc, 0.0))
            t1   = -qb - sq;  t2 = -qb + sq
            t_all = np.where(disc < 0, np.inf,
                    np.where(t1 >= 0, t1,
                    np.where(t2 >= 0, t2, np.inf)))
            t     = t_all.min(axis=1)
            hit   = t < best_t
            best_t    = np.where(hit, t, best_t)
            best_type = np.where(hit, 4, best_type)        # bullet

        # ── pack results ─────────────────────────────────────────────────────
        _TYPES = ['empty', 'wall', 'poison', 'enemy', 'bullet']
        rays = [{'type': _TYPES[int(best_type[i])],
                 'dist': round(float(best_t[i]) / FOV_RANGE, 4)}
                for i in range(N)]

        result = {'rays': rays}
        if not self.headless:
            vis_t = np.where(best_type == 1,
                             np.minimum(best_t + WALL_PEEK, FOV_RANGE),
                             best_t)
            wx = np.round(p.x + dx * vis_t, 1)
            wy = np.round(p.y + dy * vis_t, 1)
            result['cone_pts'] = [[float(wx[i]), float(wy[i])] for i in range(N)]

        return result

    def get_state(self):
        return {
            'players':      {pid: p.to_dict() for pid, p in self.players.items()},
            'bullets':      [b.to_dict() for b in self.bullets.values()],
            'poison':       self.poison.to_dict(),
            'obstacles':    [o.to_dict() for o in self.obstacles],
            'observations': {pid: self.compute_observation(pid) for pid in self.players},
        }
