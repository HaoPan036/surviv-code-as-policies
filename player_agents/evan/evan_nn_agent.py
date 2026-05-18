"""
Self-play PPO training for Surviv RL.
Two agents share one GRU actor-critic policy (symmetric self-play).

Observation  : 31 rays × [5 one-hot type + 1 dist] + 1 normalised HP  = 187-dim
Action       : 7 independent Bernoulli bits [w, s, a, d, q, e, space]
Reward (dense): (Δhp_opp − Δhp_self) / 100  per step
Reward (terminal): +1 win / −1 lose / 0 draw-or-timeout
"""

import argparse
import math
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from game import Game, MAP_W, MAP_H, PLAYER_HP

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
OBS_DIM      = 31 * 6 + 1   # 187
ACT_DIM      = 7             # w s a d q e space
ACT_KEYS     = ['w', 's', 'a', 'd', 'q', 'e', 'space']
HIDDEN       = 128

MAX_STEPS    = 2000          # steps per episode (~33 s at 60 Hz)
DT           = 1 / 60
EPS_PER_UPDATE = 10          # episodes to collect before a PPO update
PPO_EPOCHS   = 4
CLIP         = 0.2
GAMMA        = 0.99
LAM          = 0.95
ENT_COEF     = 0.01
VF_COEF      = 0.5
MAX_GRAD     = 0.5
LR           = 3e-4
TBPTT_CHUNK  = 120       # 2 seconds at 60 Hz — GRU only learns dependencies this far back



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ray type → one-hot index
_TYPE_IDX = {'empty': 0, 'wall': 1, 'poison': 2, 'enemy': 3, 'bullet': 4}

# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def encode(obs_dict, hp: float) -> torch.Tensor:
    """Convert compute_observation() dict + hp → flat float32 tensor (OBS_DIM,)."""
    if obs_dict is None:
        return torch.zeros(OBS_DIM, dtype=torch.float32)
    parts = []
    for ray in obs_dict['rays']:
        oh = [0.0] * 5
        oh[_TYPE_IDX.get(ray['type'], 0)] = 1.0
        parts.extend(oh)
        parts.append(float(ray['dist']))
    parts.append(hp / PLAYER_HP)
    return torch.tensor(parts, dtype=torch.float32)


def to_keys(action_bits: torch.Tensor) -> dict:
    """Convert 7-bit action tensor → input dict for game.set_input()."""
    return {k: bool(action_bits[i].item()) for i, k in enumerate(ACT_KEYS)}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRUActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in   = nn.Linear(OBS_DIM, HIDDEN)
        self.gru     = nn.GRU(HIDDEN, HIDDEN)        # sequence API; step-mode also supported
        self.pi_head = nn.Linear(HIDDEN, ACT_DIM)   # logits for Bernoulli
        self.vf_head = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """Single-step rollout.  x: (OBS_DIM,)  h: (HIDDEN,) → logits, value, h_new"""
        e        = torch.relu(self.fc_in(x))
        # nn.GRU expects (seq=1, batch=1, feat) and (layers=1, batch=1, feat)
        out, h_n = self.gru(e.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0))
        h_new    = h_n.squeeze(0).squeeze(0)          # back to (HIDDEN,)
        feat     = out.squeeze(0).squeeze(0)
        return self.pi_head(feat), self.vf_head(feat).squeeze(-1), h_new

    def forward_sequence(self, e_seq: torch.Tensor, h0: torch.Tensor):
        """Chunk forward pass.  e_seq: (T, HIDDEN)  h0: (HIDDEN,) → logits (T,A), values (T,), h_new (HIDDEN,)"""
        out, h_n = self.gru(e_seq.unsqueeze(1),          # (T, 1, HIDDEN)
                            h0.unsqueeze(0).unsqueeze(0)) # (1, 1, HIDDEN)
        out  = out.squeeze(1)                             # (T, HIDDEN)
        h_new = h_n.squeeze(0).squeeze(0)                # (HIDDEN,)
        return self.pi_head(out), self.vf_head(out).squeeze(-1), h_new

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(HIDDEN, device=DEVICE)

# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, last_val):
    """rewards/values/dones: python lists; last_val: float"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val  = last_val if t == len(rewards) - 1 else values[t + 1]
        not_done  = 1.0 - float(dones[t])
        delta     = rewards[t] + GAMMA * next_val * not_done - values[t]
        gae       = delta + GAMMA * LAM * not_done * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(policy: GRUActorCritic, game: Game, pid0: str, pid1: str):
    """
    Runs one episode, returns two trajectory dicts (one per agent).
    Each trajectory: obs, actions, log_probs, values, rewards, dones, hidden_init
    """
    game.reset()
    game.add_player(pid0)
    game.add_player(pid1, x=MAP_W / 2, y=MAP_H / 2)
    game.poison.reset()

    h = [policy.init_hidden(), policy.init_hidden()]
    pids = [pid0, pid1]

    traj = [{
        'obs': [], 'acts': [], 'log_probs': [], 'values': [],
        'rewards': [], 'dones': [], 'h0': h[i].clone()
    } for i in range(2)]

    step = 0
    done = False

    policy.eval()
    with torch.no_grad():
        for _ in range(MAX_STEPS):
            obs_raw = [game.compute_observation(pid) for pid in pids]
            players  = game.players

            hp    = [players[pid].hp if pid in players else 0.0 for pid in pids]
            obs_t = [encode(obs_raw[i], hp[i]).to(DEVICE) for i in range(2)]

            actions_list, lp_list, val_list = [], [], []
            for i in range(2):
                logits, val, h[i] = policy(obs_t[i], h[i])
                dist   = torch.distributions.Bernoulli(logits=logits)
                act    = dist.sample()
                lp     = dist.log_prob(act).sum()
                actions_list.append(act)
                lp_list.append(lp.item())
                val_list.append(val.item())

            for i, pid in enumerate(pids):
                if pid in game.players and game.players[pid].alive:
                    game.set_input(pid, to_keys(actions_list[i]))

            game.update(DT)

            alive    = [players[pid].alive if pid in players else False for pid in pids]
            terminal = not alive[0] or not alive[1]
            step += 1

            for i in range(2):
                j = 1 - i
                reward = 0.0
                if terminal:
                    if alive[i] and not alive[j]:
                        reward += 1.0
                    elif not alive[i] and alive[j]:
                        reward += -1.0

                traj[i]['obs'].append(obs_t[i].cpu())
                traj[i]['acts'].append(actions_list[i].cpu())
                traj[i]['log_probs'].append(lp_list[i])
                traj[i]['values'].append(val_list[i])
                traj[i]['rewards'].append(reward)
                traj[i]['dones'].append(float(terminal))

            if terminal:
                done = True
                break

    # Last value for bootstrap (0 if terminal)
    last_vals = [0.0, 0.0]
    if not done:
        with torch.no_grad():
            for i in range(2):
                obs_raw_i = game.compute_observation(pids[i])
                hp_i = game.players[pids[i]].hp if pids[i] in game.players else 0.0
                obs_t_i = encode(obs_raw_i, hp_i).to(DEVICE)
                _, lv, _ = policy(obs_t_i, h[i])
                last_vals[i] = lv.item()

    for i in range(2):
        adv, ret = compute_gae(traj[i]['rewards'], traj[i]['values'],
                               traj[i]['dones'], last_vals[i])
        traj[i]['advantages'] = adv
        traj[i]['returns']    = ret

    ep_len   = len(traj[0]['rewards'])
    ep_r     = [sum(t['rewards']) for t in traj]
    p0_alive = game.players.get(pid0) and game.players[pid0].alive
    p1_alive = game.players.get(pid1) and game.players[pid1].alive
    win_flag = done and p0_alive and not p1_alive

    return traj, ep_len, ep_r, win_flag

# ---------------------------------------------------------------------------
# PPO update — all trajectories batched together, TBPTT over 2-second chunks
# ---------------------------------------------------------------------------

def ppo_update(policy: GRUActorCritic, optimizer: optim.Optimizer,
               all_trajs: list[list[dict]]):
    """
    Processes all 2×EPS_PER_UPDATE trajectories as one (T_max, B) batch per epoch.
    Shorter trajectories are zero-padded; a boolean mask excludes padding from the loss.
    GRU is unrolled in TBPTT_CHUNK steps so backward never exceeds 2 seconds of history.
    """
    policy.train()
    flat_trajs = [t for ep in all_trajs for t in ep]   # B = 2 × EPS_PER_UPDATE
    B          = len(flat_trajs)
    lengths    = [len(t['obs']) for t in flat_trajs]
    T_max      = max(lengths)

    # Build padded tensors once — reused across epochs (only adv normalisation varies)
    obs_pad  = torch.zeros(T_max, B, OBS_DIM)
    act_pad  = torch.zeros(T_max, B, ACT_DIM)
    olp_pad  = torch.zeros(T_max, B)
    adv_pad  = torch.zeros(T_max, B)
    ret_pad  = torch.zeros(T_max, B)
    h0_batch = torch.zeros(B, HIDDEN)
    valid    = torch.zeros(T_max, B, dtype=torch.bool)

    for j, traj in enumerate(flat_trajs):
        T = lengths[j]
        obs_pad[:T, j] = torch.stack(traj['obs'])
        act_pad[:T, j] = torch.stack(traj['acts']).float()
        olp_pad[:T, j] = torch.tensor(traj['log_probs'])
        adv_j          = torch.tensor(traj['advantages'], dtype=torch.float32)
        adv_j          = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
        adv_pad[:T, j] = adv_j
        ret_pad[:T, j] = torch.tensor(traj['returns'], dtype=torch.float32)
        h0_batch[j]    = traj['h0']
        valid[:T, j]   = True

    obs_pad  = obs_pad.to(DEVICE)
    act_pad  = act_pad.to(DEVICE)
    olp_pad  = olp_pad.to(DEVICE)
    adv_pad  = adv_pad.to(DEVICE)
    ret_pad  = ret_pad.to(DEVICE)
    h0_batch = h0_batch.to(DEVICE)
    valid    = valid.to(DEVICE)
    n_valid  = valid.sum()

    total_loss = 0.0

    for _ in range(PPO_EPOCHS):
        # Encode all timesteps in one matmul: (T_max, B, OBS_DIM) → (T_max, B, HIDDEN)
        e_batch = torch.relu(policy.fc_in(obs_pad))

        # TBPTT: roll GRU in 2-second chunks, detach hidden state at each boundary
        h_t = h0_batch.unsqueeze(0)                             # (1, B, HIDDEN)
        logits_parts, values_parts = [], []
        for start in range(0, T_max, TBPTT_CHUNK):
            end      = min(start + TBPTT_CHUNK, T_max)
            out, h_t = policy.gru(e_batch[start:end], h_t)     # (chunk, B, HIDDEN)
            logits_parts.append(policy.pi_head(out))            # (chunk, B, ACT_DIM)
            values_parts.append(policy.vf_head(out).squeeze(-1))
            h_t = h_t.detach()

        logits = torch.cat(logits_parts)                        # (T_max, B, ACT_DIM)
        values = torch.cat(values_parts)                        # (T_max, B)

        dist      = torch.distributions.Bernoulli(logits=logits)
        new_lp    = dist.log_prob(act_pad).sum(-1)              # (T_max, B)
        entropy   = dist.entropy().sum(-1)                      # (T_max, B)

        ratio     = torch.exp(new_lp - olp_pad)
        pg_loss   = torch.max(-adv_pad * ratio,
                              -adv_pad * torch.clamp(ratio, 1 - CLIP, 1 + CLIP))
        vf_loss   = 0.5 * (values - ret_pad).pow(2)
        step_loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy   # (T_max, B)

        loss = (step_loss * valid).sum() / n_valid

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / PPO_EPOCHS

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(resume: str | None = None):
    policy    = GRUActorCritic().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    game      = Game(headless=True)

    PID0, PID1 = 'agent0', 'agent1'

    update_num = 0
    total_steps = 0
    win_counts  = defaultdict(int)

    if resume:
        ckpt = torch.load(resume, map_location=DEVICE, weights_only=True)
        policy.load_state_dict(ckpt['policy'])
        optimizer.load_state_dict(ckpt['optimizer'])
        update_num  = ckpt.get('update', 0)
        total_steps = ckpt.get('steps', 0)
        print(f'Resumed from {resume}  (update {update_num}, steps {total_steps})')

    print(f'Device: {DEVICE}   OBS_DIM={OBS_DIM}   ACT_DIM={ACT_DIM}   HIDDEN={HIDDEN}')
    print(f'{"update":>6}  {"steps":>8}  {"ep_len":>6}  '
          f'{"r0":>7}  {"r1":>7}  {"win%":>6}  {"loss":>8}  {"fps":>7}')

    while True:
        all_trajs = []
        ep_lens, ep_rs, wins = [], [], []
        t0 = time.time()

        for _ in range(EPS_PER_UPDATE):
            trajs, ep_len, ep_r, win = run_episode(policy, game, PID0, PID1)
            all_trajs.append(trajs)
            ep_lens.append(ep_len)
            ep_rs.append(ep_r)
            wins.append(win)
            total_steps += ep_len

        loss = ppo_update(policy, optimizer, all_trajs)
        update_num += 1

        elapsed  = time.time() - t0
        steps_ep = sum(ep_lens)
        fps      = steps_ep / elapsed
        avg_len  = steps_ep / EPS_PER_UPDATE
        avg_r0   = sum(r[0] for r in ep_rs) / EPS_PER_UPDATE
        avg_r1   = sum(r[1] for r in ep_rs) / EPS_PER_UPDATE
        win_pct  = 100.0 * sum(wins) / EPS_PER_UPDATE

        print(f'{update_num:>6}  {total_steps:>8}  {avg_len:>6.0f}  '
              f'{avg_r0:>7.3f}  {avg_r1:>7.3f}  {win_pct:>5.1f}%  '
              f'{loss:>8.4f}  {fps:>7.0f}')

        if update_num % 50 == 0:
            path = f'check_pts/ckpt_{update_num:05d}.pt'
            torch.save({'policy': policy.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'update': update_num,
                        'steps': total_steps}, path)
            print(f'  → saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(resume=args.resume)
