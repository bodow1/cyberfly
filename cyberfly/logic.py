import numpy as np

from .hdl import Netlist


class Evaluator:
    def __init__(self, nl: Netlist):
        self.nl = nl
        self.qnets = [q for _, q, _ in nl.dffs]
        idx = {"0": 0, "1": 1}
        driven = {o for o, _ in nl.topo}
        for q in self.qnets:
            idx.setdefault(q, len(idx))
        for _, ins in nl.topo:
            for i in ins:
                if not isinstance(i, str) and i not in driven:
                    idx.setdefault(i, len(idx))
        for o, _ in nl.topo:
            idx.setdefault(o, len(idx))
        for d, _, _ in nl.dffs:
            idx.setdefault(d, len(idx))
        for bits in nl.names.values():
            for b in bits:
                if not isinstance(b, str):
                    idx.setdefault(b, len(idx))
        self.idx = idx
        level = {}
        by_level: dict[int, list] = {}
        for o, ins in nl.topo:
            level[o] = 1 + max(level.get(i, 0) for i in ins)
            by_level.setdefault(level[o], []).append((o, ins))
        self.levels = []
        for lv in sorted(by_level):
            g = by_level[lv]
            self.levels.append((
                np.array([idx[o] for o, _ in g]),
                np.array([idx[ins[0]] for _, ins in g]),
                np.array([idx[ins[1]] if len(ins) > 1 else 0 for _, ins in g]),
            ))
        self.q = np.array([idx[q] for q in self.qnets])
        self.d = np.array([idx[d] for d, _, _ in nl.dffs])
        self.depth = len(self.levels)
        self.forced: dict[int, int] = {}
        self.values = np.zeros(len(idx), np.uint8)

    def state(self, st: dict) -> np.ndarray:
        return np.array([st[q] for q in self.qnets], np.uint8)

    def force(self, nets, value: int):
        for i, b in enumerate(nets):
            if not isinstance(b, str):
                self.forced[self.idx[b]] = (value >> i) & 1

    def force_bus(self, name: str, value: int):
        self.force(self.nl.names[name], value)

    def eval(self, qv: np.ndarray) -> np.ndarray:
        v = np.zeros(len(self.idx), np.uint8)
        v[1] = 1
        v[self.q] = qv
        for k, b in self.forced.items():
            v[k] = b
        for out, a, b in self.levels:
            v[out] = 1 - (v[a] | v[b])
        self.values = v
        return v[self.d].copy()

    def word(self, nets) -> int:
        r = 0
        for i, b in enumerate(nets):
            if b == "1":
                r |= 1 << i
            elif b != "0":
                r |= int(self.values[self.idx[b]]) << i
        return r

    def bus(self, name: str) -> int:
        return self.word(self.nl.names[name])


class GateMachine:
    def __init__(self, nl: Netlist, program: list[int], words: int = 256):
        self.nl = nl
        self.ev = Evaluator(nl)
        self.ram = nl.rams[0] if nl.rams else None
        self.mem = list(program) + [0] * (words - len(program))
        st = nl.init_state()
        if self.ram is None:
            for i, w in enumerate(program):
                for b, net in enumerate(nl.names.get(f"core.mem[{i}]", [])):
                    if not isinstance(net, str):
                        st[net] = (w >> b) & 1
        self.qv = self.ev.state(st)
        self.qi = {q: i for i, q in enumerate(self.ev.qnets)}
        self.io_in = 0

    def latch(self, net) -> int:
        return int(self.qv[self.qi[net]])

    def latched(self, name: str) -> int:
        return sum(self.latch(b) << i for i, b in enumerate(self.nl.names[name]) if not isinstance(b, str))

    def step(self):
        ev = self.ev
        ev.force(self.nl.port_bits, self.io_in)
        if self.ram is None:
            self.qv = ev.eval(self.qv)
            return
        r = self.ram
        ev.force(r["rdata"], 0)
        ev.eval(self.qv)
        addr = ev.word(r["addr"])
        ev.force(r["rdata"], self.mem[addr])
        nq = ev.eval(self.qv)
        if ev.word([r["we"]]):
            self.mem[addr] = ev.word(r["wdata"])
        self.qv = nq
