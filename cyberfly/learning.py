from dataclasses import dataclass

import numpy as np

from .lif import KICK, PARAMS, Sim

RATE_HZ = 150.0
TRIAL_STEPS = 5000
ETA = 0.05
BASELINE = 0.3
COST = 0.2
PRE_CAP, POST_CAP = 10, 3
EXPLORE_HZ = 2.0
READER_GAIN = 40.0
READER_MAX = 250
READER_INIT = 30
PIXEL_FAN, PIXEL_WEIGHT = 8, 45
CLASS_GAIN, WTA_WEIGHT, WTA_VETO = 130, 242, -800


def features(frames: np.ndarray, empty: np.ndarray) -> np.ndarray:
    b = np.asarray(frames, np.float32)
    dark = np.clip((empty - b - 0.05) / 0.4, 0, 1).reshape(len(b), -1)
    bright = np.clip((b - empty - 0.05) / 0.3, 0, 1).reshape(len(b), -1)
    return np.concatenate([dark, bright, np.ones((len(b), 1), np.float32)], axis=1)


def train_reader(F: np.ndarray, y: np.ndarray, classes: int, epochs: int = 3000, lr: float = 0.5,
                 l2: float = 1e-4) -> np.ndarray:
    W = np.random.default_rng(0).normal(0, 0.01, (F.shape[1], classes))
    Y = np.eye(classes)[y]
    for _ in range(epochs):
        z = F @ W
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        W -= lr * (F.T @ ((p - Y) / len(F)) + l2 * W)
    scale = 3 * 162 / np.percentile(np.abs(F @ W).max(1), 90)
    return np.round(W * scale).astype(np.int64)


@dataclass
class Plastic:
    edges: np.ndarray
    pre: np.ndarray
    gain: np.ndarray
    depress_only: np.ndarray
    wmax: np.ndarray


class CallCircuit:
    def __init__(self, fly: tuple, visual: tuple[list, list], kc: list, call: list, reader: np.ndarray,
                 eye_size: int = 721, reader_init: int = READER_INIT, seed: int = 7):
        n, pre, post, count = fly
        d, nc = reader.shape
        self.call = np.asarray(call)
        self.pixel, self.twin = n, n + d
        self.cls = self.twin + d
        self.wta = self.cls + nc
        self.decision = self.wta + nc
        self.size = self.decision + nc
        bp, bq, bc = [], [], []
        rng = np.random.default_rng(seed)
        for j in range(d - 1):
            pool = visual[(j % (2 * eye_size)) // eye_size]
            for v in rng.choice(pool, min(PIXEL_FAN, len(pool)), replace=False):
                bp.append(self.pixel + j), bq.append(int(v)), bc.append(PIXEL_WEIGHT)
        for j, c in zip(*np.nonzero(reader)):
            src = self.pixel + j if reader[j, c] > 0 else self.twin + j
            bp.append(src), bq.append(self.cls + c), bc.append(int(reader[j, c]))
        for j in range(nc):
            bp += [self.cls + j, self.decision + j]
            bq += [self.decision + j, self.wta + j]
            bc += [CLASS_GAIN, WTA_WEIGHT]
            for k in range(nc):
                if k != j:
                    bp += [self.wta + j, self.wta + j]
                    bq += [self.cls + k, self.decision + k]
                    bc += [WTA_VETO, WTA_VETO]
            for v in self.call:
                bp.append(self.decision + j), bq.append(int(v)), bc.append(reader_init)
        self.sim = Sim(self.size, np.concatenate([pre, bp]), np.concatenate([post, bq]),
                       np.concatenate([count, bc]))
        self.plastic = self._plastic(np.asarray(kc), np.arange(self.decision, self.decision + nc))

    def _edges(self, sources: np.ndarray) -> np.ndarray:
        s = self.sim
        targets = set(self.call.tolist())
        return np.array([e for i in sources for e in range(s.indptr[i], s.indptr[i + 1]) if s.post[e] in targets],
                        np.int64)

    def _plastic(self, kc: np.ndarray, decisions: np.ndarray) -> Plastic:
        ekc, erd = self._edges(kc), self._edges(decisions)
        edges = np.concatenate([ekc, erd])
        w0 = self.sim.w[edges]
        w_syn = PARAMS["w_syn"]
        wmax = np.maximum(w0 * 4, w_syn * 400)
        wmax[len(ekc):] = w_syn * READER_MAX
        return Plastic(
            edges=edges,
            pre=np.searchsorted(self.sim.indptr, edges, side="right") - 1,
            gain=np.concatenate([np.ones(len(ekc)), np.full(len(erd), READER_GAIN)]).astype(np.float32),
            depress_only=np.arange(len(edges)) < len(ekc),
            wmax=wmax.astype(np.float32),
        )

    def look(self, f: np.ndarray, seed: int, explore_hz: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        steps, pix = np.nonzero(rng.random((TRIAL_STEPS, len(f))) < f * RATE_HZ * PARAMS["dt"] * 1e-3)
        ns = rng.poisson(explore_hz * len(self.call) * TRIAL_STEPS * PARAMS["dt"] * 1e-3)
        kt = np.concatenate([steps, steps, rng.integers(0, TRIAL_STEPS, ns)])
        ki = np.concatenate([self.pixel + pix, self.twin + pix, rng.choice(self.call, ns)])
        self.sim.reset()
        _, i = self.sim.run(TRIAL_STEPS, (kt.astype(np.int64), ki.astype(np.int32), np.full(len(kt), KICK, np.float32)))
        return np.bincount(i, minlength=self.size)

    def calls(self, spikes: np.ndarray) -> int:
        return int(spikes[self.call].sum())

    def reinforce(self, spikes: np.ndarray, reward: float, eta: float = ETA, baseline: float = BASELINE,
                  cost: float = COST) -> float:
        post = self.calls(spikes)
        if not post:
            return 0.0
        r = reward - baseline - cost
        p = self.plastic
        dw = eta * p.gain * r * np.minimum(spikes[p.pre], PRE_CAP) * min(post, POST_CAP) * PARAMS["w_syn"]
        dw[p.depress_only] = np.minimum(dw[p.depress_only], 0)
        self.sim.w[p.edges] = np.clip(self.sim.w[p.edges] + dw.astype(np.float32), 0, p.wmax)
        return r

    def call_rate(self, F: np.ndarray, seed: int = 50_000) -> float:
        return float(np.mean([self.calls(self.look(f, seed + k)) >= 1 for k, f in enumerate(F)]))
