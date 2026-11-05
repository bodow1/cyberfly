import argparse
import json
import time
from pathlib import Path

import numpy as np

from cyberfly.brain import Brain, shooter_wiring
from cyberfly.compiler import compile_fpu
from cyberfly.connectome import Connectome
from cyberfly.firmware import build
from cyberfly.hdl import Netlist
from cyberfly.logic import GateMachine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=25_000)
    ap.add_argument("--out", type=Path, default=Path("runs/shooter"))
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    program = build("shooter.c")
    nl = Netlist.build("fly_fpu_gpu")
    fpu = compile_fpu(nl)
    fpu.load(program)
    fly = Connectome.load()
    brain = Brain(fpu, fly, shooter_wiring(fpu, fly), gpu=args.gpu)
    print(brain.stats, flush=True)

    reference = GateMachine(nl, program)
    flops = [q for _, q, _ in nl.dffs]
    ports = nl.port_bits
    nf = len(fly)
    dn = {cell: name for cell, name in zip(brain.wiring.inputs, ("DNa02_L", "DNa02_R", "GF_L", "GF_R"))}
    record = np.zeros(brain.n, np.uint8)
    record[list(dn)] = 1
    for q in flops + ports:
        record[nf + fpu.latch_neuron(q)] = 1
    even = [p for p in fpu.input_samples if p % 2 == 0]
    input_phase = min(even) if even else fpu.input_samples[0] - 1
    io_wr = nl.names["io_wr"][0]
    dist = nl.names["dist"]

    mismatches, frames, dn_spikes = 0, [], []
    start, done = time.time(), 0
    while done < args.cycles:
        n = min(5, args.cycles - done)
        t, i = brain.run(n * brain.cycle_steps, record=record)
        spikes = set(zip(t.tolist(), i.tolist()))
        dn_spikes += [(int(s), dn[c]) for s, c in zip(t.tolist(), i.tolist()) if c in dn]
        for cycle in range(done, done + n):
            state = brain.states(spikes, flops, cycle)
            mismatches += sum(state[q] != reference.latch(q) for q in flops)
            if state[io_wr]:
                frames.append(dict(
                    cycle=cycle,
                    status=sum(state[b] << k for k, b in enumerate(nl.names["io_out"])),
                    demon=sum(state[b] << k for k, b in enumerate(nl.names["frame_demon"])),
                    dist=[sum(state[dist[c * 5 + j]] << j for j in range(5)) for c in range(32)],
                ))
            ts = brain.time(input_phase + cycle * fpu.M)
            reference.io_in = sum(int((ts, nf + fpu.latch_neuron(q)) in spikes) << b for b, q in enumerate(ports))
            reference.step()
        done += n
        print(f"cycle {done}/{args.cycles}  {time.time() - start:.0f}s  frames {len(frames)}  mismatches {mismatches}",
              flush=True)
        if done % 100 == 0 or done >= args.cycles:
            (args.out / "run.json").write_text(json.dumps(dict(
                cycles=done, mismatches=mismatches, frames=frames, descending=dn_spikes, M=fpu.M,
                stats=brain.stats)))
        if frames and frames[-1]["status"] >> 31:
            break


if __name__ == "__main__":
    main()
