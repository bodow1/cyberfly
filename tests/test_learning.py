import numpy as np
import pytest

from cyberfly.learning import CallCircuit, features, train_reader

EYE = 6


def toy_fly(seed: int = 0):
    rng = np.random.default_rng(seed)
    visual = (list(range(0, 4)), list(range(4, 8)))
    kc = list(range(8, 38))
    call = [38]
    pre, post, count = [], [], []
    for k in kc:
        for v in rng.choice(8, 3, replace=False):
            pre.append(v), post.append(k), count.append(60)
        pre.append(k), post.append(call[0]), count.append(1)
    return (39, np.array(pre), np.array(post), np.array(count)), visual, kc, call


def frames(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.full((n, 2, EYE), 0.6, np.float32)
    x[y == 0, :, :3] = 0.1
    x[y == 1, :, 3:] = 0.1
    return x + rng.normal(0, 0.03, x.shape).astype(np.float32), y


def train(shuffle: bool, trials: int = 120):
    fly, visual, kc, call = toy_fly()
    x, y = frames(200, 1)
    empty = np.full((2, EYE), 0.6, np.float32)
    F = features(x, empty)
    circuit = CallCircuit(fly, visual, kc, call, train_reader(F[:150], y[:150], 2), eye_size=EYE, reader_init=8)
    test = F[150:][y[150:] == 0]
    before = circuit.call_rate(test)
    rng = np.random.default_rng(3)
    for t in range(trials):
        k = int(rng.integers(0, 150))
        spikes = circuit.look(F[k], t, explore_hz=2.0)
        pays = rng.random() < 0.5 if shuffle else y[k] == 0
        circuit.reinforce(spikes, float(pays))
    return before, circuit.call_rate(test), circuit.call_rate(F[150:][y[150:] == 1])


def test_call_rate_rises_with_reward():
    before, after, other = train(shuffle=False)
    assert after >= before + 0.5 and other <= 0.2


def test_call_rate_flat_with_shuffled_reward():
    before, after, _ = train(shuffle=True)
    assert after <= before + 0.1
