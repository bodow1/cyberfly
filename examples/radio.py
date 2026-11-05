import argparse
import json
import time
from pathlib import Path

import numpy as np

from cyberfly.compiler import compile_fpu
from cyberfly.firmware import build
from cyberfly.hdl import Netlist
from cyberfly.lif import KICK
from cyberfly.llm import ask, compile_program, finished, to_asm
from cyberfly.radio import BITS, ORIGIN, SLOT, add_radio, decode, frame_spikes, program_frames, schedule
from cyberfly.runtime import NeuralFPU, period


def window(kicks: list[tuple[int, int]], t0: int, t1: int):
    sel = [(t, n) for t, n in kicks if t0 <= t < t1]
    if not sel:
        return None
    t, n = zip(*sel)
    return np.array(t, np.int64), np.array(n, np.int32), np.full(len(t), KICK, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default="walk a square, then rest")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--after", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("runs/radio"))
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nl = Netlist.build("fly_fpu")
    fpu = compile_fpu(nl)
    boot = build("boot.c")
    fpu.load(boot)
    radio = add_radio(fpu)
    neural = NeuralFPU(fpu, gpu=args.gpu)
    neural.record[[radio.tx, radio.rx]] = 1
    pc, io = nl.names["pc"], nl.names["io_out"]
    nets = [q for q in pc + io if q in fpu.latches]
    for q in nets:
        neural.record[fpu.latch_neuron(q)] = 1
    p, steps = period(), neural.cycle_steps
    print(f"FPU {len(fpu.circuit.kind)} neurons, transmit neuron {radio.tx}, receive neuron {radio.rx}", flush=True)

    log = dict(request=args.request, cycles=[], tx=[], rx=[], frames=[], program=None, asm=None)
    pending, asked, jumped, cycle, start = [], None, None, 0, time.time()
    while jumped is None or cycle < jumped + args.after:
        t0 = neural.sim.t
        st, si = neural.spikes(1, window(pending, t0, t0 + steps))
        log["tx"] += st[si == radio.tx].tolist()
        log["rx"] += st[si == radio.rx].tolist()
        spikes = set(zip(st.tolist(), si.tolist()))
        state = neural.states(spikes, nets, cycle)
        pcv = sum(state.get(q, 0) << i for i, q in enumerate(pc))
        iov = sum(state.get(q, 0) << i for i, q in enumerate(io))
        log["cycles"].append((cycle, pcv, iov))
        if asked is None:
            heard = decode(log["tx"], p)
            if heard and neural.sim.t > heard[0][0] + SLOT * (BITS + 1) * p:
                asked = heard[0]
                print(f"cycle {cycle}: the FPU transmits {chr(asked[1])!r}; asking the LLM", flush=True)
                program = ask(args.request, offline=args.offline or None)
                words = compile_program(program)
                frames = program_frames(words)
                gaps = schedule(boot, frames)
                log.update(program=program, asm=to_asm(program))
                t = neural.sim.t + 2 * steps
                for k, (w, g) in enumerate(zip(frames, gaps)):
                    pending += [(int(s) - 1, radio.rx) for s in frame_spikes(w, t, p)]
                    log["frames"].append(dict(k=k, word=w, step=t))
                    t += g * steps
                print(f"{json.dumps(program)}\n{4 * len(words)} bytes, {len(frames)} frames, "
                      f"{sum(gaps)} cycles", flush=True)
        if jumped is None and pcv >= ORIGIN:
            jumped = cycle
            print(f"cycle {cycle}: the FPU jumps to the received program at {pcv:#x}", flush=True)
        if cycle % 100 == 0:
            print(f"cycle {cycle}  pc {pcv:#05x}  io_out {iov:#010x}  wall {time.time() - start:.0f} s", flush=True)
        if jumped is not None and finished(iov):
            print(f"cycle {cycle}: the program has finished", flush=True)
            break
        cycle += 1
    (args.out / "run.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
