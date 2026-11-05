from enum import IntEnum

import numpy as np

W = 242
FANOUT = 400


class Kind(IntEnum):
    CLOCK = 0
    GATE = 1
    LATCH_A = 2
    LATCH_B = 3
    RING = 4
    OSC = 5
    CAPTURE = 6
    FOLLOW = 7


class Circuit:
    def __init__(self, M: int):
        assert M % 2 == 0 and M >= 4
        self.M = M
        self.kind: list[int] = []
        self.phase: list[int] = []
        self.name: list[str] = []
        self.pre: list[int] = []
        self.post: list[int] = []
        self.cnt: list[int] = []
        self.fanout: list[int] = []
        self.ring = [self.new(Kind.RING, k, f"R{k}") for k in range(M)]
        for k in range(M):
            self.edge(self.ring[k - 1], self.ring[k], W)
        self.clocks = {k: [self.ring[k]] for k in range(M)}
        self.even = [self.new(Kind.OSC, 0, "E0")]
        self.odd = [self.new(Kind.OSC, 1, "O0")]
        self.edge(self.even[0], self.odd[0], W)
        self.edge(self.odd[0], self.even[0], W)
        self.latches: list[tuple[int, int]] = []
        self.boot: list[tuple[int, int]] = []

    def new(self, kind: Kind, phase: int, name: str = "") -> int:
        self.kind.append(int(kind))
        self.phase.append(phase % self.M)
        self.name.append(name)
        self.fanout.append(0)
        return len(self.kind) - 1

    def edge(self, a: int, b: int, c: int):
        self.pre.append(a)
        self.post.append(b)
        self.cnt.append(c)
        self.fanout[a] += 1

    def clock(self, k: int, growth: bool = False) -> int:
        k %= self.M
        limit = FANOUT if growth else FANOUT - 1
        pool = self.clocks[k]
        for n in pool:
            if self.fanout[n] < limit:
                return n
        src = self.clock(k - 1, growth=True)
        n = self.new(Kind.CLOCK, k, f"C{k}.{len(pool)}")
        self.edge(src, n, W)
        pool.append(n)
        return n

    def oscillator(self, parity: int, growth: bool = False) -> int:
        limit = FANOUT if growth else FANOUT - 1
        pool = self.even if parity == 0 else self.odd
        for n in pool:
            if self.fanout[n] < limit:
                return n
        src = self.oscillator(1 - parity, growth=True)
        n = self.new(Kind.OSC, parity, f"{'EO'[parity]}{len(pool)}")
        self.edge(src, n, W)
        pool.append(n)
        return n

    def gate(self, k: int, name: str = "") -> int:
        g = self.new(Kind.GATE, k, name)
        self.edge(self.clock(k - 1), g, W)
        return g

    def nor(self, inputs: list[int], k: int, lags: list[int] | None = None) -> int:
        g = self.gate(k)
        for i, lag in zip(inputs, lags or [1] * len(inputs)):
            assert lag in (1, 2) and self.is_at(i, k - lag)
            self.edge(i, g, -W)
        return g

    def is_at(self, n: int, k: int) -> bool:
        k %= self.M
        if self.kind[n] == Kind.LATCH_A:
            return k % 2 == 0
        if self.kind[n] == Kind.LATCH_B:
            return k % 2 == 1
        return self.phase[n] == k

    def latch(self, init: int = 0, name: str = "") -> int:
        a = self.new(Kind.LATCH_A, 0, name + ".A")
        b = self.new(Kind.LATCH_B, 1, name + ".B")
        self.edge(self.oscillator(1), a, W)
        self.edge(self.oscillator(0), b, W)
        self.edge(b, a, -W)
        self.edge(a, b, -W)
        self.latches.append((a, b))
        self.boot.append((0, a) if init else (1, b))
        return len(self.latches) - 1

    def latch_set(self, li: int, s: int):
        self.edge(s, self.latches[li][1], -W)

    def latch_reset(self, li: int, r: int):
        self.edge(r, self.latches[li][0], -W)

    def arrays(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        return (len(self.kind), np.array(self.pre, np.int64), np.array(self.post, np.int64),
                np.array(self.cnt, np.int64))

    def boot_kicks(self) -> list[tuple[int, int]]:
        k = [(0, self.ring[0])]
        for ph, pool in self.clocks.items():
            k += [(ph, n) for n in pool if n != self.ring[0] and self.kind[n] == Kind.CLOCK]
        k += [(0, n) for n in self.even] + [(1, n) for n in self.odd]
        return k + self.boot

    def stats(self) -> dict:
        k = np.array(self.kind)
        counts = {kind.name.lower(): int((k == kind).sum()) for kind in Kind}
        return counts | {"neurons": len(k), "connections": len(self.cnt),
                         "synapses": int(np.abs(np.array(self.cnt)).sum())}
