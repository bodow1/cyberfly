from dataclasses import dataclass, field

import numpy as np

from .compiler import FPU
from .iss import ISS
from .neurons import W, Kind
from .ports import acknowledge, copy_into_port

BITS = 14
MASK = (1 << BITS) - 1
SLOT = 6
SEND, ACK, VALID = 31, 30, 31
YBAR_EXCITE, YBAR_VETO = 363, 484
MAGIC = 3
ORIGIN = 0x200
MAX_WORDS = 96


@dataclass
class Radio:
    tx: int
    rx: int


def add_radio(fpu: FPU) -> Radio:
    return Radio(tx=_transmitter(fpu), rx=_receiver(fpu))


def _not(fpu: FPU, net) -> int:
    return fpu.circuit.latches[fpu.latches[net][0]][1]


def _transmitter(fpu: FPU) -> int:
    c, names = fpu.circuit, fpu.netlist.names
    io, wr = names["io_out"], names["io_wr"][0]
    antenna = c.new(Kind.FOLLOW, 0, "tx")
    t0 = fpu.sample_phase + 2
    assert t0 + SLOT * BITS < fpu.first_write - 1
    slots = [(t0, [])] + [(t0 + SLOT * (j + 1), [_not(fpu, io[j])]) for j in range(BITS)]
    for k, vetoes in slots:
        h = c.gate(k, f"tx@{k}")
        for v in [_not(fpu, wr), _not(fpu, io[SEND])] + vetoes:
            c.edge(v, h, -W)
        c.edge(h, antenna, W)
    return antenna


def _relay(c, name: str) -> int:
    return c.new(Kind.FOLLOW, 0, name)


def _receiver(fpu: FPU) -> int:
    c, nl = fpu.circuit, fpu.netlist
    rx = _relay(c, "rx")
    start = _relay(c, "rx.start")
    c.edge(rx, start, W)
    taps = [start]
    for m in range(1, SLOT * (BITS + 1) + 8):
        n = _relay(c, f"rx.tap{m}")
        c.edge(taps[-1], n, W)
        taps.append(n)
    hold = [c.latch(0, f"rx.bit{j}") for j in range(BITS)]
    valid = c.latch(0, "rx.valid")
    for j in range(BITS):
        slot = SLOT * (j + 1)
        c.edge(taps[slot - 1], start, -W)
        ybar = _relay(c, f"rx.ybar{j}")
        c.edge(taps[slot - 1], ybar, YBAR_EXCITE)
        c.edge(rx, ybar, -YBAR_VETO)
        seen = _relay(c, f"rx.seen{j}")
        c.edge(taps[slot + 1], seen, W)
        c.edge(ybar, seen, -2 * W)
        echo = _relay(c, f"rx.echo{j}")
        c.edge(seen, echo, W)
        for src in (seen, echo):
            s = c.new(Kind.GATE, 0, f"rx.set{j}")
            c.edge(src, s, W)
            c.latch_set(hold[j], s)
    for m in (SLOT * BITS + 5, SLOT * BITS + 6):
        s = c.new(Kind.GATE, 0, "rx.setvalid")
        c.edge(taps[m], s, W)
        c.latch_set(valid, s)

    acknowledge(fpu, ACK, hold + [valid], "rx")
    copy_into_port(fpu, dict(enumerate(hold)) | {VALID: valid}, "rx")
    return rx


def frame_spikes(word: int, start: int, period: int) -> np.ndarray:
    return np.array([start] + [start + SLOT * (j + 1) * period for j in range(BITS) if (word >> j) & 1], np.int64)


def decode(times, period: int) -> list[tuple[int, int]]:
    times = sorted(int(t) for t in times)
    frames, i = [], 0
    while i < len(times):
        t0, word = times[i], 0
        end = t0 + (SLOT * BITS + SLOT // 2) * period
        for t in times[i + 1:]:
            if t > end:
                break
            j = round((t - t0) / (SLOT * period)) - 1
            if 0 <= j < BITS and abs(t - t0 - SLOT * (j + 1) * period) <= period:
                word |= 1 << j
        frames.append((t0, word))
        while i < len(times) and times[i] <= end:
            i += 1
    return frames


def program_frames(words: list[int]) -> list[int]:
    assert len(words) <= MAX_WORDS
    frames, total = [MAGIC << 12 | len(words)], 0
    for w in words:
        parts = [w & 0x7FF, (w >> 11) & 0x7FF, (w >> 22) & 0x3FF]
        frames += parts
        total += sum(parts)
    return frames + [total & MASK]


def ask_cycle(boot: list[int], limit: int = 100_000) -> int:
    iss = ISS(boot)
    while not any((v >> SEND) & 1 for v in iss.io_out):
        assert iss.steps < limit
        iss.step()
    return 2 * iss.steps


def schedule(boot: list[int], frames: list[int], margin: int = 6, first: int = 60) -> list[int]:
    queue, acks, state = list(frames), [], {"low": False}

    def load():
        if state["low"]:
            state["low"] = False
            return 0
        return (1 << VALID | queue[0]) if queue else 0

    def store(v):
        if (v >> ACK) & 1 and not (v >> SEND) & 1:
            acks.append(iss.steps)
            state["low"] = True
            if queue:
                queue.pop(0)

    iss = ISS(boot, io_in=load, on_store=store)
    while len(acks) < len(frames):
        assert iss.steps < 1_000_000
        iss.step()
    gaps = [2 * (b - a) + margin for a, b in zip(acks, acks[1:])] + [margin]
    gaps[0] += first
    return gaps


@dataclass
class Link:
    arrivals: list[tuple[float, int]]
    ack_latency: float = 2.5
    word: int = 0
    valid: bool = False
    clear_at: float | None = None
    overlaps: int = 0
    sent: list[int] = field(default_factory=list)

    @classmethod
    def paced(cls, frames: list[int], gaps: list[int], start: float, jitter: float = 0.5, seed: int = 0, **kw):
        rng = np.random.default_rng(seed)
        arrivals, t = [], start
        for w, g in zip(frames, gaps):
            arrivals.append((t + rng.uniform(0, jitter), w))
            t += g
        return cls(arrivals, **kw)

    def load(self, cycle: float) -> int:
        if self.clear_at is not None and cycle >= self.clear_at:
            while self.arrivals and self.arrivals[0][0] <= self.clear_at:
                self.arrivals.pop(0)
                self.overlaps += 1
            self.word, self.valid, self.clear_at = 0, False, None
        while self.arrivals and self.arrivals[0][0] <= cycle:
            self.overlaps += self.valid
            self.word |= self.arrivals.pop(0)[1]
            self.valid = True
        return (1 << VALID | self.word) if self.valid else 0

    def store(self, cycle: float, value: int):
        if (value >> SEND) & 1:
            self.sent.append(value & MASK)
        elif (value >> ACK) & 1:
            self.clear_at = cycle + self.ack_latency
