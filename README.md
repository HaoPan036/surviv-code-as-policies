# surviv-code-as-policies

A hand-written **Code-as-Policies** agent for a Surviv.io-style 2D shooter,
built as a course project. No training, no neural net — the controller is a
small program written by hand, composed of named subroutines and selected by
priority. The agent (`player_agents/hao.py`) competes against trained NN and
other hand-coded baselines in 1v1 and battle-royale modes.

## My contribution

| Path | Author |
|---|---|
| `player_agents/hao.py` | **Mine.** The hand-coded "Code-as-Policies" agent. |
| `player_agents/__init__.py` | Mine (empty package marker). |
| Everything else (`game.py`, `server.py`, `watch.py`, `static/`, `player_agents/evan/`, …) | Course material, distributed by the instructor. Not authored by me. |

## The game in one paragraph

Top-down 2D shooter on a 1000×1000 map. 60 Hz tick. Players move at 200 u/s,
turn at 150°/s, fire every 0.35 s (4 hits to kill at 25 dmg / 100 HP). View
is a 60° cone reaching 288 units; each frame the engine returns a **31-ray
observation** with `{type ∈ {empty, wall, poison, enemy, bullet}, dist 0..1}`
per ray. A shrinking poison zone (radius 720 → 0 over ~140 s after a 20 s
grace) forces engagement. Action keys: `w/s/a/d/q/e/space`.

## Agent interface (restricted)

The instructor's harness wraps `game` in a `_RestrictedGame` proxy — agents
may only access `game.poison` and the per-step observation; touching
`game.players`, `game.obstacles`, etc. raises `AttributeError` and silently
no-ops the agent for that frame. `hao.py` was written to that restriction
from the start (it only ever reads `game.poison.{cx, cy, radius}`).

```python
class HaoPolicy:
    def __init__(self): ...
    def step(self, obs_raw, player, game) -> dict[str, bool]:
        # returns {'w','s','a','d','q','e','space'} → bool
```

## Strategy

Decision priority, highest first:

```
P1  RUN_HOME       poison_dist + 50 u margin > radius  →  sprint to centre
P2  DODGE_BULLET   bullet ray dist < 0.35              →  perpendicular strafe
P3  ENGAGE / RETREAT
    ├─ ≥ 2 enemy clusters AND not late-game  →  multi-retreat (no fire)
    └─ 1 enemy OR poison radius < 200        →  _engage(...)
        ├─ EXEC mode (enemy ≥ 5 frames, 0 bullets ≥ 25 frames)
        │            →  charge head-on, sustained fire, no kite
        └─ regular   →  bearing-rate lead with warmup, sustained fire,
                        random-walk kite, point-blank engage distance
P4  PATROL         no enemy in cone  →  sweep ± 180°, drift toward centre
```

Key design decisions and why:

- **`ENGAGE_DIST = 0.22` (≈ 63 units, point-blank).** A trained NN's main edge
  is *leading the shot*. Lead becomes useless when bullet flight time is
  shorter than the target's reaction. At 0.22 the bullet arrives in ≈ 0.10 s
  and a player at 200 u/s only moves ≈ 20 units — about one body radius.
  Sweep across this parameter: 0.40 → 28 %, 0.30 → 31 %, **0.22 → 43 %**,
  0.15 → 37 % (collision-bound).
- **EXEC mode.** A trained NN fires at any visible enemy. So if we *see* an
  enemy for ≥ 25 frames and *no incoming bullet* appears for ≥ 25 frames,
  they almost certainly don't see us. Stop kiting, charge head-on, dump
  rounds. `EXEC_QUIET_T > NN's FIRE_CD (21 frames)` prevents false triggers
  during the NN's reload window.
- **Random-walk kite.** Drift-matching kite (strafe with the enemy's
  apparent motion) exposes a clean bearing-rate signal that the NN's GRU
  can lead. Replacing it with 8–18-frame fully random pulses denies the NN
  a stable signal.
- **BR-aware retreat.** Reward goes to the *last survivor*, not the most
  kills. When ≥ 2 distinct enemies are visible we slip away toward zone
  centre; let them whittle each other down. Adjacent enemy rays count as
  one body (gap > 3 indices = different player).
- **Late-game override (`poison.radius < 200`).** Eventually camping costs
  more than fighting; unlock engagement on every enemy in the final ring.
- **Sustained fire (`FIRE_TOL_EXTRA = 2`).** Widen the firing tolerance
  beyond the strict aim tolerance — don't waste a single `FIRE_CD` window
  while mid-correcting.

## Results

| Setup | Score |
|---|---|
| 1v1 vs Evan NN (`ckpt_05800.pt`, 70 M training steps) | **43 % decisive wins** |
| 1v1 vs Evan `CodedPolicy` (hand-coded baseline) | ~50 % (tied) |
| 6-agent royale, 50 episodes (`Evan NN`, `Evan Code`, `GE`, `SM`, `Hcwang`, `Hao`) | **9 wins / avg place 3.14** (tied #1) |

## What I tried that didn't work (notes to self)

- **Ambush behind a wall** — 60° FOV is too narrow to camp safely; getting
  pinned with a wall in front blinds you to flanking enemies. 12 W / 39 L
  vs NN.
- **Lead without warmup** — bearing-rate from a single frame is pure noise
  and pushes the aim index past the firing tolerance every step → agent
  never pulls the trigger. EMA + ≥ 4 stable frames before trusting it.
- **Drift-matching kite** — predictable to a trained model. Random is better
  here than any "smart" heuristic.
- **Early-game total camp** (radius > 500 → flee from every enemy) — the
  field's NNs steal all the kills while you hide; you survive but win less.
- **Low-HP cover retreat** — the user wanted aggression in 1v1 vs NN;
  retreating at HP < 20 just delays death.

## Repository layout

```
.
├── game.py                   # game engine                       (course)
├── server.py                 # human-play web server             (course)
├── watch.py                  # spectator / eval server           (course)
├── static/                   # web frontend                       (course)
├── player_agents/
│   ├── __init__.py
│   ├── evan/                 # baseline + trained NN              (course)
│   │   ├── evan_code_agent.py
│   │   ├── evan_nn_agent.py
│   │   └── ckpt_05800.pt
│   └── hao.py                # the hand-coded policy             ← MINE
└── requirements.txt
```

## Running it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch 'numpy<2'

# Spectate two policies fighting:
python watch.py player_agents/evan/ckpt_05800.pt
# → open http://localhost:8766
```

To pit my agent against the NN: in the watch UI, set one slot to *Coded*
and import `HaoPolicy` from `player_agents.hao` in place of the default
`CodedPolicy` (or wire it into the registry if you have the battle-royale
version of `watch.py`).

## Background reading

- Liang et al., **Code as Policies** (Google, 2023) — the agent-as-program
  framing that inspired this project. The LLM is the programmer; the
  controller is just code. <https://code-as-policies.github.io/>
