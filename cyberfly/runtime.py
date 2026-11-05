from functools import cache

import numpy as np

from .compiler import FPU
from .lif import KICK, Sim, simulator
from .neurons import Circuit


@cache
def period() -> int:
    c = Circuit(4)
    sim = Sim(*c.arrays())
    t, i = sim.run(400, (np.array([0]), np.array([c.ring[0]]), np.array([KICK], np.float32)))
    d = np.diff(t[np.isin(i, c.ring)])
    assert (d == d[0]).all()
    return int(d[0])


def boot_kicks(circuit: Circuit, offset: int = 0, t0: int = 1):
    p = period()
    kicks = circuit.boot_kicks()
    steps = np.array([t0 + ph * p - 1 for ph, _ in kicks], np.int64)
    ids = np.array([offset + n for _, n in kicks], np.int32)
    return steps, ids, np.full(len(kicks), KICK, np.float32)


class Clocked:
    t0 = 1

    def __init__(self, fpu: FPU, offset: int = 0):
        self.fpu = fpu
        self.offset = offset
        self.cycle_steps = fpu.M * period()

    def time(self, phase: int) -> int:
        return self.t0 + phase * period()

    def sample_time(self, cycle: int) -> int:
        return self.time(self.fpu.sample_phase + cycle * self.fpu.M)

    def states(self, spikes: set, nets: list, cycle: int) -> dict:
        t = self.sample_time(cycle)
        return {q: int((t, self.offset + self.fpu.latch_neuron(q)) in spikes) for q in nets}


class NeuralFPU(Clocked):
    def __init__(self, fpu: FPU, gpu: bool = False):
        super().__init__(fpu)
        self.sim = simulator(*fpu.circuit.arrays(), gpu=gpu)
        self.kicks = boot_kicks(fpu.circuit)
        self.record = np.zeros(len(fpu.circuit.kind), np.uint8)
        nl = fpu.netlist
        self.watch = [q for _, q, _ in nl.dffs if q in fpu.latches]
        self.memory = {wb: fpu.circuit.latches[lis[0]][0] for wb, lis in fpu.memory.items()}
        for q in self.watch:
            self.record[fpu.latch_neuron(q)] = 1
        for n in self.memory.values():
            self.record[n] = 1

    def run(self, cycles: int, kicks=None):
        t, i = self.spikes(cycles, kicks)
        return set(zip(t.tolist(), i.tolist()))

    def spikes(self, cycles: int, kicks=None):
        k = ([self.kicks] if self.sim.t == 0 else []) + ([kicks] if kicks is not None else [])
        merged = tuple(np.concatenate(x) for x in zip(*k)) if k else None
        return self.sim.run(cycles * self.cycle_steps, merged, record=self.record)

    def ram(self, spikes: set, cycle: int) -> dict:
        t = self.sample_time(cycle)
        return {wb: int((t, n) in spikes) for wb, n in self.memory.items()}
