from dataclasses import dataclass, field

import numpy as np

from .compiler import FPU
from .connectome import Connectome
from .lif import simulator
from .neurons import W
from .runtime import Clocked, boot_kicks

LC10C = ["LC10c-1", "LC10c-2"]


@dataclass
class Wiring:
    outputs: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)


def split(cells: list[int], k: int, rng: np.random.Generator) -> list[list[int]]:
    cells = list(rng.permutation(cells))
    return [cells[i::k] for i in range(k)]


def descending(fly: Connectome) -> dict:
    one = lambda t, s: fly.cells([t], s)[0]
    return {"DNa02_L": one("DNa02", "left"), "DNa02_R": one("DNa02", "right"),
            "GF_L": one("DNp01", "left"), "GF_R": one("DNp01", "right")}


def shooter_wiring(fpu: FPU, fly: Connectome, seed: int = 0) -> Wiring:
    rng = np.random.default_rng(seed)
    left = split(fly.cells(LC10C, "left"), 16, rng)
    right = split(fly.cells(LC10C, "right"), 16, rng)
    lc4 = split(fly.cells(["LC4"]), 32, rng)
    pixels = fpu.netlist.names["frame_demon"]
    outputs = {f"c{c}": (pixels[c], (left[c] if c < 16 else right[c - 16]) + lc4[c]) for c in range(32)}
    dn = descending(fly)
    inputs = {dn["DNa02_L"]: 0, dn["DNa02_R"]: 1, dn["GF_L"]: 2, dn["GF_R"]: 2}
    return Wiring(outputs, inputs)


def steering_wiring(fpu: FPU, fly: Connectome, seed: int = 0) -> Wiring:
    rng = np.random.default_rng(seed)
    io = fpu.netlist.names["io_out"]
    groups = (split(fly.cells(LC10C, "left"), 4, rng) + split(fly.cells(LC10C, "right"), 4, rng)
              + split(fly.cells(["LC4"]), 4, rng))
    outputs = {f"o{b}": (io[b], groups[b]) for b in range(12)}
    dn = descending(fly)
    inputs = {dn["DNa02_L"]: 0, dn["DNa02_R"]: 1, dn["GF_L"]: 2, dn["GF_R"]: 2}
    return Wiring(outputs, inputs)


class Brain(Clocked):
    def __init__(self, fpu: FPU, fly: Connectome, wiring: Wiring, extra: int = 0, extra_edges=None,
                 gpu: bool = False):
        nf = len(fly)
        super().__init__(fpu, offset=nf)
        self.fly = fly
        self.wiring = wiring
        followers = fpu.follow({name: net for name, (net, _) in wiring.outputs.items()})
        n_cpu, cpre, cpost, ccnt = fpu.circuit.arrays()
        pre, post = [], []
        for name, (_, cells) in wiring.outputs.items():
            pre += [nf + followers[name]] * len(cells)
            post += cells
        for cell, bit in wiring.inputs.items():
            pre.append(cell)
            post.append(nf + fpu.inputs[bit])
        bpre, bpost, bcnt = np.array(pre), np.array(post), np.full(len(pre), W)
        self.extra = nf + n_cpu
        self.n = nf + n_cpu + extra
        parts = [(fly.pre, fly.post, fly.count), (cpre + nf, cpost + nf, ccnt), (bpre, bpost, bcnt)]
        if extra_edges is not None:
            parts.append(extra_edges)
        self.sim = simulator(self.n, *(np.concatenate(x) for x in zip(*parts)), gpu=gpu)
        self.kicks = boot_kicks(fpu.circuit, offset=nf)
        self.stats = dict(fly=nf, fpu=n_cpu, bridge=len(pre), extra=extra, total=self.n, M=fpu.M)

    def run(self, steps: int, kicks=None, record=None):
        k = [self.kicks] if self.sim.t == 0 else []
        if kicks is not None:
            k.append(kicks)
        merged = tuple(np.concatenate(x) for x in zip(*k)) if k else None
        return self.sim.run(steps, merged, record=record)
