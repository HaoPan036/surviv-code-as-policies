"""Spectate two agents playing against each other.

Usage:
    python watch.py ckpt_02700.pt
    python watch.py ckpt_02700.pt --port 8766
"""
import argparse
import time
import threading

import torch
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from game import Game, MAP_W, MAP_H
from player_agents.evan.evan_nn_agent import GRUActorCritic, encode, to_keys, DEVICE
from player_agents.evan.evan_code_agent import CodedPolicy

PID0, PID1 = 'agent0', 'agent1'

app      = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'watch'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

game     = Game(headless=False)
policy   = None
_h       = [None, None]
_episode = 0
_step    = 0

_policy_types = ['nn', 'nn']       # 'nn' or 'code', one per agent
_coded        = [CodedPolicy(), CodedPolicy()]
_started      = threading.Event()  # set when the frontend sends set_policies
_wins         = [0, 0]             # wins for agent0, agent1
_draws        = 0


def _load_policy(ckpt_path: str):
    global policy
    policy = GRUActorCritic().to(DEVICE)
    ckpt   = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    print(f'Loaded {ckpt_path}  '
          f'(update {ckpt.get("update", "?")}, steps {ckpt.get("steps", "?")})')


def _reset_episode():
    global _h, _step
    game.reset()
    game.add_player(PID0)
    game.add_player(PID1, x=MAP_W / 2, y=MAP_H / 2)
    _h        = [policy.init_hidden(), policy.init_hidden()]
    _coded[0] = CodedPolicy()
    _coded[1] = CodedPolicy()
    _step     = 0


def _step_agent(i: int, pid: str):
    p = game.players.get(pid)
    if not p or not p.alive:
        return
    obs_raw = game.compute_observation(pid)
    if _policy_types[i] == 'nn':
        obs_t = encode(obs_raw, p.hp).to(DEVICE)
        logits, _, _h[i] = policy(obs_t, _h[i])
        act = torch.distributions.Bernoulli(logits=logits).sample()
        game.set_input(pid, to_keys(act))
    else:
        game.set_input(pid, _coded[i].step(obs_raw, p, game))


def _game_loop():
    global _episode, _draws
    _started.wait()      # block until the frontend picks policies and clicks Start
    _reset_episode()
    last = time.time()

    while True:
        now  = time.time()
        dt   = min(now - last, 0.05)
        last = now

        with torch.no_grad():
            for i, pid in enumerate([PID0, PID1]):
                _step_agent(i, pid)

        game.update(dt)
        global _step
        _step += 1

        state                 = game.get_state()
        state['episode']      = _episode
        state['step']         = _step
        state['policy_types'] = _policy_types[:]
        state['wins']         = _wins[:]
        state['draws']        = _draws
        socketio.emit('state', state)

        p0      = game.players.get(PID0)
        p1      = game.players.get(PID1)
        ep_over = not (p0 and p0.alive) or not (p1 and p1.alive)
        if ep_over:
            a0_alive = p0 and p0.alive
            a1_alive = p1 and p1.alive
            if a0_alive and not a1_alive:
                _wins[0] += 1
            elif a1_alive and not a0_alive:
                _wins[1] += 1
            else:
                _draws += 1
            _episode += 1
            _reset_episode()

        time.sleep(1 / 60)


@app.route('/')
def index():
    return send_from_directory('static', 'watch.html')


@socketio.on('set_policies')
def on_set_policies(data):
    global _draws
    _policy_types[0] = 'nn' if data.get('p0') == 'nn' else 'code'
    _policy_types[1] = 'nn' if data.get('p1') == 'nn' else 'code'
    _wins[0] = _wins[1] = _draws = 0
    _started.set()      # first call unblocks the game loop
    _reset_episode()    # subsequent calls just restart the episode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to .pt checkpoint file')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()

    _load_policy(args.checkpoint)
    threading.Thread(target=_game_loop, daemon=True).start()
    print(f'Spectator view → http://localhost:{args.port}')
    socketio.run(app, host='0.0.0.0', port=args.port,
                 debug=False, allow_unsafe_werkzeug=True)
