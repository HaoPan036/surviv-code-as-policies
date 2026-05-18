import argparse
import time
import threading
import uuid

import torch
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from game import Game, MAP_W, MAP_H
from player_agents.evan.evan_nn_agent import GRUActorCritic, encode, to_keys, DEVICE

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'surviv'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

game = Game(headless=False)
_sid_to_pid: dict[str, str] = {}

BOT_PREFIX = 'bot_'
_policy: GRUActorCritic | None = None
_bot_hs: dict[str, torch.Tensor] = {}


def _load_policy(ckpt_path: str):
    global _policy
    _policy = GRUActorCritic().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    _policy.load_state_dict(ckpt['policy'])
    _policy.eval()
    print(f'Loaded {ckpt_path}  '
          f'(update {ckpt.get("update", "?")}, steps {ckpt.get("steps", "?")})')


def _reset_bots(n: int):
    global _bot_hs
    for pid in [p for p in game.players if p.startswith(BOT_PREFIX)]:
        game.remove_player(pid)
    _bot_hs = {}
    for i in range(n):
        bid = f'{BOT_PREFIX}{i}'
        game.add_player(bid)
        if _policy is not None:
            _bot_hs[bid] = _policy.init_hidden()


def _ai_bot_step():
    for bid in list(_bot_hs):
        if bid not in game.players:
            continue
        p = game.players[bid]
        if not p.alive:
            continue
        obs_raw = game.compute_observation(bid)
        obs_t = encode(obs_raw, p.hp).to(DEVICE)
        with torch.no_grad():
            logits, _, _bot_hs[bid] = _policy(obs_t, _bot_hs[bid])
        act = torch.distributions.Bernoulli(logits=logits).sample()
        game.set_input(bid, to_keys(act))


def _game_result() -> str | None:
    """'win' if all bots dead with a human still alive, else None."""
    bot_ids = [bid for bid in _bot_hs if bid in game.players]
    if not bot_ids:
        return None
    if any(game.players[bid].alive for bid in bot_ids):
        return None
    human_pids = [pid for pid in _sid_to_pid.values() if pid in game.players]
    if not human_pids:
        return None
    return 'win' if any(game.players[pid].alive for pid in human_pids) else None


def _game_loop():
    last = time.time()
    while True:
        now = time.time()
        dt = min(now - last, 0.05)
        last = now

        _ai_bot_step()
        game.update(dt)
        state = game.get_state()
        result = _game_result()
        if result:
            state['result'] = result
        socketio.emit('state', state)
        time.sleep(1 / 60)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@socketio.on('connect')
def on_connect():
    pid = uuid.uuid4().hex[:8]
    _sid_to_pid[request.sid] = pid
    bot_ids = [bid for bid in _bot_hs if bid in game.players]
    game_over = not bot_ids or not any(game.players[bid].alive for bid in bot_ids)
    if len(_sid_to_pid) == 1 or game_over:  # first human, or rejoining a finished game
        n = max(1, min(int(request.args.get('bots', 1)), 4))
        game.reset()
        _reset_bots(n)
    game.add_player(pid)
    emit('welcome', {'id': pid})
    print(f'[+] {pid} joined  ({len(_sid_to_pid)} online)')


@socketio.on('disconnect')
def on_disconnect():
    pid = _sid_to_pid.pop(request.sid, None)
    if pid:
        game.remove_player(pid)
        print(f'[-] {pid} left  ({len(_sid_to_pid)} online)')


@socketio.on('keys')
def on_keys(data):
    pid = _sid_to_pid.get(request.sid)
    if pid:
        game.set_input(pid, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', default='check_pts/ckpt_01950.pt',
                        help='Path to .pt checkpoint (default: check_pts/ckpt_01950.pt)')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    _load_policy(args.checkpoint)
    game.reset()
    _reset_bots(1)
    threading.Thread(target=_game_loop, daemon=True).start()
    print(f'Starting server on http://localhost:{args.port}')
    socketio.run(app, host='0.0.0.0', port=args.port, debug=False, allow_unsafe_werkzeug=True)
