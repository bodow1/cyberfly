from dataclasses import dataclass

import numpy as np

from .compiler import FPU
from .lif import KICK, PARAMS, Sim
from .neurons import W
from .ports import held_inputs
from .runtime import Clocked, boot_kicks

RATE_HZ = 150.0
TRIAL_STEPS = 5000
GAP_STEPS = 2000
ETA = 0.05
BASELINE = 0.3
PRE_CAP, POST_CAP = 10, 3
EXPLORE_HZ = 2.0
READER_GAIN = 40.0
READER_MAX = 250
READER_INIT = 30
PIXEL_FAN, PIXEL_WEIGHT = 8, 45
CLASS_GAIN, WTA_WEIGHT, WTA_VETO = 130, 242, -800
SIGN_BIT = 4
SETTLE_CYCLES = 3
MAX_RESPONSE_CYCLES = 40


@dataclass
class Tool:
    name: str
    call: np.ndarray
    dans: np.ndarray
    call_bit: int
    reward_bit: int
    ack_bit: int
    cost: float


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
    W_ = np.random.default_rng(0).normal(0, 0.01, (F.shape[1], classes))
    Y = np.eye(classes)[y]
    for _ in range(epochs):
        z = F @ W_
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        W_ -= lr * (F.T @ ((p - Y) / len(F)) + l2 * W_)
    scale = 3 * 162 / np.percentile(np.abs(F @ W_).max(1), 90)
    return np.round(W_ * scale).astype(np.int64)


class CallCircuit:
    def __init__(self, fly: tuple, visual: tuple[list, list], kc: list, tools: list[Tool], reader: np.ndarray,
                 fpu: FPU | None = None, eye_size: int = 721, reader_init: int = READER_INIT, seed: int = 7):
        n, pre, post, count = fly
        d, nc = reader.shape
        self.tools = tools
        bp, bq, bc = [], [], []

        def syn(a, b, w):
            bp.append(a), bq.append(b), bc.append(w)

        self.fpu, self.clock, offset = fpu, None, n
        if fpu is not None:
            io = fpu.netlist.names["io_out"]
            entries = {}
            for t in tools:
                entries |= held_inputs(fpu, [t.call_bit], t.ack_bit, f"call.{t.name}")
            followers = fpu.follow({t.name: io[t.reward_bit] for t in tools})
            for t in tools:
                for cell in t.call:
                    syn(int(cell), n + entries[t.call_bit], W)
                for cell in t.dans:
                    syn(n + followers[t.name], int(cell), W)
            names = fpu.netlist.names
            self.watch = [names["io_wr"][0]] + [names["io_out"][t.ack_bit] for t in tools]
            m, fpre, fpost, fcount = fpu.circuit.arrays()
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
                syn(self.decision + j, n + fpu.inputs[SIGN_BIT + j], W)
        parts = [(pre, post, count)]
        if fpu is not None:
            parts.append((fpre + n, fpost + n, fcount))
        parts.append((np.array(bp), np.array(bq), np.array(bc)))
        self.sim = Sim(self.size, *(np.concatenate(x) for x in zip(*parts)))
        decisions = np.arange(self.decision, self.decision + nc)
        self.plastic = {t.name: self._plastic(t.call, np.asarray(kc), decisions) for t in tools}
        self.reference = {t.name: 1.0 for t in tools}
        self.pending = boot_kicks(fpu.circuit, offset=n) if fpu is not None else None

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

    def _kicks(self, f, steps: int, rng, explore_hz: float, extra=None):
        t0 = self.sim.t
        k, pix = np.nonzero(rng.random((steps, len(f))) < f * RATE_HZ * PARAMS["dt"] * 1e-3)
        calls = np.concatenate([t.call for t in self.tools])
        ns = rng.poisson(explore_hz * len(calls) * steps * PARAMS["dt"] * 1e-3)
        kt = [k + t0, k + t0, rng.integers(0, steps, ns) + t0]
        ki = [self.pixel + pix, self.twin + pix, rng.choice(calls, ns)]
        for src in [self.pending, extra]:
            if src is not None:
                m = (src[0] >= t0) & (src[0] < t0 + steps)
                kt.append(src[0][m]), ki.append(src[1][m])
        kt, ki = np.concatenate(kt).astype(np.int64), np.concatenate(ki).astype(np.int32)
        return kt, ki, np.full(len(kt), KICK, np.float32)

    def _run(self, f, steps, rng, explore_hz=0.0, record=None, extra=None):
        return self.sim.run(steps, self._kicks(f, steps, rng, explore_hz, extra), record=record)

    def look(self, f: np.ndarray, seed: int, explore_hz: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.fpu is None:
            self.sim.reset()
        _, i = self._run(f, TRIAL_STEPS, rng, explore_hz)
        return np.bincount(i, minlength=self.size)

    def calls(self, spikes: np.ndarray, tool: Tool) -> int:
        return int(spikes[tool.call].sum())

    def respond(self, f: np.ndarray, seed: int, called: set, extra=None, steps: int = GAP_STEPS) -> dict:
        rng = np.random.default_rng(seed + 1_000_000)
        record = np.zeros(self.size, np.uint8)
        for t in self.tools:
            record[t.dans] = 1
        dans = {t.name: 0 for t in self.tools}

        def run(image, n):
            st, si = self._run(image, n, rng, record=record, extra=extra)
            counts = np.bincount(si, minlength=self.size)
            for t in self.tools:
                dans[t.name] += int(counts[t.dans].sum())
            return st, si

        if self.fpu is None:
            run(f, steps)
            return dans
        clock, watch = self.clock, self.watch
        for q in watch:
            record[clock.offset + self.fpu.latch_neuron(q)] = 1
        cycle = (self.sim.t - clock.sample_time(0)) // clock.cycle_steps + 1
        pending, image = set(called), f
        for _ in range(MAX_RESPONSE_CYCLES):
            if not pending:
                break
            st, si = run(image, clock.sample_time(cycle) + 1 - self.sim.t)
            state = clock.states(set(zip(st.tolist(), si.tolist())), watch, cycle)
            if state[watch[0]]:
                pending -= {t.name for t, q in zip(self.tools, watch[1:]) if state[q]}
            cycle += 1
        run(np.zeros_like(f), SETTLE_CYCLES * clock.cycle_steps + GAP_STEPS)
        return dans

    def settle(self, f: np.ndarray, steps: int = GAP_STEPS):
        self._run(np.zeros_like(f), steps, np.random.default_rng(self.sim.t), record=np.zeros(self.size, np.uint8))

    def reinforce(self, spikes: np.ndarray, dans: dict, eta: float = ETA, baseline: float = BASELINE) -> dict:
        out = {}
        for t in self.tools:
            post = self.calls(spikes, t)
            if not post:
                continue
            r = dans[t.name] / self.reference[t.name] - baseline - t.cost
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
            self.settle(f)
        if self.fpu is not None:
            self.settle(F[0], MAX_RESPONSE_CYCLES * self.clock.cycle_steps)
        return {name: float(np.mean(h)) for name, h in hits.items()}
