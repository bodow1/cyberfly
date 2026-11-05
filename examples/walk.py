import argparse
import json
import time
from pathlib import Path

import numpy as np

from cyberfly.body import OMMATIDIA, STEPS_PER_EXCHANGE, Eyes, Motor, calibrate, darkness, world
from cyberfly.brain import Brain, steering_wiring
from cyberfly.compiler import compile_fpu
from cyberfly.connectome import Connectome
from cyberfly.firmware import build
from cyberfly.hdl import Netlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="zigzag.c")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--boot-cycles", type=int, default=28)
    ap.add_argument("--out", type=Path, default=Path("runs/walk"))
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fpu = compile_fpu(Netlist.fpu())
    fpu.load(build(args.program))
    fly = Connectome.load()
    wiring = steering_wiring(fpu, fly)
    eyes = Eyes.wire(fly, first=len(fly) + len(fpu.circuit.kind) + len(wiring.outputs))
    brain = Brain(fpu, fly, wiring, extra=2 * OMMATIDIA, extra_edges=eyes.edges, gpu=args.gpu)
    print(brain.stats, flush=True)
    brain.run(args.boot_cycles * brain.cycle_steps)

    groups = {"DNa02_L": fly.cells(["DNa02"], "left"), "DNa02_R": fly.cells(["DNa02"], "right"),
              "GF": fly.cells(["DNp01"])}
    owner = {c: k for k, cells in groups.items() for c in cells}
    record = np.zeros(brain.n, np.uint8)
    record[list(owner)] = 1
    io_out = fpu.netlist.names["io_out"]
    for q in io_out:
        record[len(fly) + fpu.latch_neuron(q)] = 1

    sim, camera = world()
    obs, cal, sky = calibrate(sim)
    motor = Motor()
    rng = np.random.default_rng(0)
    path, out = [], 0
    start = time.time()
    for chunk in range(int(args.seconds / 0.01)):
        dark = darkness(obs["vision"], cal, sky)
        t, i = brain.run(STEPS_PER_EXCHANGE, eyes.kicks(dark, brain.sim.t, rng), record=record)
        counts = {k: 0 for k in groups}
        for c in i.tolist():
            if c in owner:
                counts[owner[c]] += 1
        motor.update(counts, {k: len(v) for k, v in groups.items()}, 0.01)
        action = motor.action()
        for _ in range(STEPS_PER_EXCHANGE):
            obs, *_ = sim.step(action)
            sim.render()
        cycle = (brain.sim.t - 1 - brain.sample_time(0)) // brain.cycle_steps
        if brain.sample_time(cycle) >= brain.sim.t - STEPS_PER_EXCHANGE:
            state = brain.states(set(zip(t.tolist(), i.tolist())), io_out, cycle)
            out = sum(state[q] << b for b, q in enumerate(io_out))
        path.append(dict(t=chunk * 0.01, xy=[float(v) for v in obs["fly"][0][:2]], io_out=out,
                         **{k: round(v, 1) for k, v in motor.rates.items()}))
        if chunk % 100 == 0:
            print(f"t={chunk * 0.01:.1f}s  wall {time.time() - start:.0f}s  io_out {out:#05x}  "
                  f"DNa02 L {motor.rates['DNa02_L']:.0f} R {motor.rates['DNa02_R']:.0f} Hz", flush=True)
    camera.save_video(args.out / "walk.mp4")
    (args.out / "path.json").write_text(json.dumps(path))


if __name__ == "__main__":
    main()
