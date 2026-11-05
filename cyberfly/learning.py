from dataclasses import dataclass

import numpy as np

from .compiler import FPU
from .lif import KICK, PARAMS, Sim
from .neurons import W
from .ports import acknowledge, held_call, held_choice
from .runtime import Clocked, boot_kicks

RATE_HZ = 150.0
TRIAL_STEPS = 5000
GAP_STEPS = 2000
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
SIGN_BIT = 4
SETTLE_STEPS = 200
MAX_RESPONSE_CYCLES = 80


@dataclass
class Tool:
    name: str
    call: np.ndarray
    dans: np.ndarray
    call_bit: int
    ack_bit: int
    reward_bit: int
    reads: tuple


@dataclass
class Plastic:
    edges: np.ndarray
    pre: np.ndarray
    gain: np.ndarray
    depress_only: np.ndarray
    wmax: np.ndarray


def features(frames: np.ndarray, empty: np.ndarray) -> np.ndarray:
    b = np.asarray(frames, np.float32)
    dark = np.clip((empty - b - 0.05) / 0.4, 0, 1).reshape(len(b), -1)
    bright = np.clip((b - empty - 0.05) / 0.3, 0, 1).reshape(len(b), -1)
    return np.concatenate([dark, bright, np.ones((len(b), 1), np.float32)], axis=1)


def train_reader(F: np.ndarray, y: np.ndarray, classes: int, epochs: int = 3000, lr: float = 0.5,
                 l2: float = 1e-4) -> np.ndarray:
    w = np.random.default_rng(0).normal(0, 0.01, (F.shape[1], classes))
    Y = np.eye(classes)[y]
    for _ in range(epochs):
        z = F @ w
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        w -= lr * (F.T @ ((p - Y) / len(F)) + l2 * w)
    scale = 3 * 162 / np.percentile(np.abs(F @ w).max(1), 90)
    return np.round(w * scale).astype(np.int64)


class CallCircuit:
    def __init__(self, fly: tuple, visual: tuple[list, list], kc: list, tools: list[Tool], reader: np.ndarray,
                 fpu: FPU | None = None, eye_size: int = 721, reader_init: int = READER_INIT, seed: int = 7):
        n, pre, post, count = fly
        d, nc = reader.shape
        self.tools, self.fpu, self.clock = tools, fpu, None
        bp, bq, bc = [], [], []

        def syn(a, b, w):
            bp.append(a), bq.append(b), bc.append(w)

        offset, parts = n, [(pre, post, count)]
        if fpu is not None:
            choice, held = held_choice(fpu, [SIGN_BIT + j for j in range(nc)], "sign")
            calls = {t.name: held_call(fpu, t.call_bit, f"call.{t.name}") for t in tools}
            for t in tools:
                acknowledge(fpu, t.ack_bit, [calls[t.name][1]] + [held[b] for b in t.reads], f"ack.{t.name}")
            io = fpu.netlist.names["io_out"]
            followers = fpu.follow({f"reward.{t.name}": io[t.reward_bit] for t in tools})
            for t in tools:
                for cell in t.call:
                    syn(int(cell), n + calls[t.name][0], W)
                for cell in t.dans:
                    syn(n + followers[f"reward.{t.name}"], int(cell), W)
            self.choice = {bit: n + e for bit, e in choice.items()}
            self.entries = {t.name: n + calls[t.name][0] for t in tools}
            self.busy = [n + fpu.circuit.latches[li][0] for _, li in calls.values()]
            self.busy += [n + fpu.latch_neuron(io[t.reward_bit]) for t in tools]
            m, fpre, fpost, fcount = fpu.circuit.arrays()
            parts.append((fpre + n, fpost + n, fcount))
            offset = n + m
            self.clock = Clocked(fpu, offset=n)
        self.pixel, self.twin = offset, offset + d
        self.cls = self.twin + d
        self.wta = self.cls + nc
        self.decision = self.wta + nc
        self.size = self.decision + nc
        rng = np.random.default_rng(seed)
        for j in range(d - 1):
            pool = visual[(j % (2 * eye_size)) // eye_size]
            for v in rng.choice(pool, min(PIXEL_FAN, len(pool)), replace=False):
                syn(self.pixel + j, int(v), PIXEL_WEIGHT)
        for j, c in zip(*np.nonzero(reader)):
            syn(self.pixel + j if reader[j, c] > 0 else self.twin + j, self.cls + c, int(reader[j, c]))
        for j in range(nc):
            syn(self.cls + j, self.decision + j, CLASS_GAIN)
            syn(self.decision + j, self.wta + j, WTA_WEIGHT)
            for k in range(nc):
                if k != j:
                    syn(self.wta + j, self.cls + k, WTA_VETO)
                    syn(self.wta + j, self.decision + k, WTA_VETO)
            for t in tools:
                for v in t.call:
                    syn(self.decision + j, int(v), reader_init)
            if fpu is not None:
                syn(self.decision + j, self.choice[SIGN_BIT + j], W)
        parts.append((np.array(bp), np.array(bq), np.array(bc)))
        self.sim = Sim(self.size, *(np.concatenate(x) for x in zip(*parts)))
        decisions = np.arange(self.decision, self.decision + nc)
        self.plastic = {t.name: self._plastic(t.call, np.asarray(kc), decisions) for t in tools}
        self.reference = {t.name: 1.0 for t in tools}
        self.pending = boot_kicks(fpu.circuit, offset=n) if fpu is not None else None
        self.brain = np.ones(self.size, np.uint8)
        if fpu is not None:
            self.brain[n:offset] = 0
        self.dans = np.zeros(self.size, np.uint8)
        for t in tools:
            self.dans[t.dans] = 1

    def _edges(self, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
        s, targets = self.sim, set(targets.tolist())
        return np.array([e for i in sources for e in range(s.indptr[i], s.indptr[i + 1]) if s.post[e] in targets],
                        np.int64)

    def _plastic(self, call: np.ndarray, kc: np.ndarray, decisions: np.ndarray) -> Plastic:
        ekc, erd = self._edges(kc, call), self._edges(decisions, call)
        edges = np.concatenate([ekc, erd])
        w_syn = PARAMS["w_syn"]
        wmax = np.maximum(self.sim.w[edges] * 4, w_syn * 400)
        wmax[len(ekc):] = w_syn * READER_MAX
        return Plastic(
            edges=edges,
            pre=np.searchsorted(self.sim.indptr, edges, side="right") - 1,
            gain=np.concatenate([np.ones(len(ekc)), np.full(len(erd), READER_GAIN)]).astype(np.float32),
            depress_only=np.arange(len(edges)) < len(ekc),
            wmax=wmax.astype(np.float32),
        )

    def _run(self, f, steps: int, rng, record, explore_hz: float = 0.0, extra=None):
        t0 = self.sim.t
        k, pix = np.nonzero(rng.random((steps, len(f))) < f * RATE_HZ * PARAMS["dt"] * 1e-3)
        calls = np.concatenate([t.call for t in self.tools])
        ns = rng.poisson(explore_hz * len(calls) * steps * PARAMS["dt"] * 1e-3)
        kt = [k + t0, k + t0, rng.integers(0, steps, ns) + t0]
        ki = [self.pixel + pix, self.twin + pix, rng.choice(calls, ns)]
        for src in (self.pending, extra):
            if src is not None:
                m = (src[0] >= t0) & (src[0] < t0 + steps)
                kt.append(src[0][m]), ki.append(src[1][m])
        kt, ki = np.concatenate(kt).astype(np.int64), np.concatenate(ki).astype(np.int32)
        return self.sim.run(steps, (kt, ki, np.full(len(kt), KICK, np.float32)), record=record)

    def look(self, f: np.ndarray, seed: int, explore_hz: float = 0.0) -> np.ndarray:
        if self.fpu is None:
            self.sim.reset()
        else:
            self.sim.quiet(self.brain)
        _, i = self._run(f, TRIAL_STEPS, np.random.default_rng(seed), self.brain, explore_hz)
        return np.bincount(i, minlength=self.size)

    def calls(self, spikes: np.ndarray, tool: Tool) -> int:
        return int(spikes[tool.call].sum())

    def respond(self, seed: int, extra=None) -> np.ndarray:
        rng = np.random.default_rng(seed + 1_000_000)
        blank = np.zeros(self.twin - self.pixel)
        record = self.dans.copy()
        counts = np.zeros(self.size, np.int64)
        if self.fpu is None:
            _, i = self._run(blank, GAP_STEPS, rng, record, extra=extra)
            return counts + np.bincount(i, minlength=self.size)
        clock = self.clock
        record[self.busy] = 1
        cycle = (self.sim.t + SETTLE_STEPS - clock.sample_time(0)) // clock.cycle_steps + 1
        for _ in range(MAX_RESPONSE_CYCLES):
            t = clock.sample_time(cycle)
            st, si = self._run(blank, t + 1 - self.sim.t, rng, record, extra=extra)
            counts += np.bincount(si, minlength=self.size)
            if not np.isin(self.busy, si[st == t]).any():
                break
            cycle += 1
        _, si = self._run(blank, SETTLE_STEPS, rng, record, extra=extra)
        return counts + np.bincount(si, minlength=self.size)

    def trial(self, f: np.ndarray, seed: int, explore_hz: float = EXPLORE_HZ, extra=None) -> tuple[np.ndarray, dict]:
        spikes = self.look(f, seed, explore_hz)
        after = self.respond(seed, extra)
        dans = {t.name: int(spikes[t.dans].sum() + after[t.dans].sum()) for t in self.tools}
        return spikes, dans

    def probe(self, tool: Tool, bit: int, seed: int = 0) -> int:
        t0 = self.sim.t + 10
        kicks = (np.array([t0, t0]), np.array([self.entries[tool.name], self.choice[bit]]))
        after = self.respond(seed, extra=kicks)
        return int(after[tool.dans].sum())

    def reinforce(self, spikes: np.ndarray, dans: dict, eta: float = ETA, baseline: float = BASELINE,
                  cost: float = COST) -> dict:
        out = {}
        for t in self.tools:
            post = self.calls(spikes, t)
            if not post:
                continue
            r = dans[t.name] / self.reference[t.name] - baseline - cost
            p = self.plastic[t.name]
            dw = eta * p.gain * r * np.minimum(spikes[p.pre], PRE_CAP) * min(post, POST_CAP) * PARAMS["w_syn"]
            dw[p.depress_only] = np.minimum(dw[p.depress_only], 0)
            self.sim.w[p.edges] = np.clip(self.sim.w[p.edges] + dw.astype(np.float32), 0, p.wmax)
            out[t.name] = r
        return out

    def call_rates(self, F: np.ndarray, seed: int = 50_000) -> dict:
        hits = {t.name: [] for t in self.tools}
        for k, f in enumerate(F):
            spikes = self.look(f, seed + k)
            for t in self.tools:
                hits[t.name].append(self.calls(spikes, t) >= 1)
            if self.fpu is not None:
                self.respond(seed + k)
        return {name: float(np.mean(h)) for name, h in hits.items()}
