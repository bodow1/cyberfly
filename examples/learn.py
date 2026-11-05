import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cyberfly.compiler import compile_fpu
from cyberfly.connectome import Connectome
from cyberfly.firmware import build
from cyberfly.hdl import Netlist
from cyberfly.learning import SIGN_BIT, CallCircuit, Tool, features, train_reader

SIGNS = Path(__file__).resolve().parent.parent / "data" / "signs.npz"
GROUP_OF = {"left": "arrow", "right": "arrow", "letter": "letter", "stone": "stone", "blank": "blank", "none": "none"}
GROUPS = {"arrow": 0.3, "letter": 0.3, "stone": 0.15, "blank": 0.1, "none": 0.15}
VISUAL = ("visual_projection", "optic", "visual_centrifugal")


def mushroom_body(fly: Connectome):
    a = fly.annotations
    kc = a[a.cell_class == "Kenyon_Cell"]["index"].to_numpy()
    onto = np.isin(fly.post, kc)
    drive = pd.Series(np.abs(fly.count[onto]), index=fly.pre[onto]).groupby(level=0).sum()
    info = a.set_index("index")
    visual = [i for i in drive.index if info.super_class.get(i) in VISUAL and drive[i] > 20]
    left = [i for i in visual if info.side.get(i) == "left"] or visual
    right = [i for i in visual if info.side.get(i) == "right"] or visual
    return (left, right), kc


def tools(fly: Connectome, kinds: list[str], names: list[str]) -> list[Tool]:
    cells = lambda types: np.array(fly.cells(types))
    bit = lambda kind: SIGN_BIT + kinds.index(kind)
    every = [
        Tool("cpu", cells(["MBON09"]), cells(["PAM12", "PAM13", "PAM14"]), 3, 30, 12, (bit("left"), bit("right"))),
        Tool("llm", cells(["MBON27"]), cells(["PAM01"]), 10, 29, 13, (bit("letter"),)),
    ]
    return [t for t in every if t.name in names]


def synapses(circuit: CallCircuit) -> dict:
    out = {}
    for name, p in circuit.plastic.items():
        reader = ~p.depress_only
        out[f"{name}.pre"] = np.where(reader, p.pre - circuit.decision, p.pre)
        out[f"{name}.post"] = circuit.sim.post[p.edges]
        out[f"{name}.reader"] = reader
        out[f"{name}.w"] = circuit.sim.w[p.edges].copy()
    return out


def transfer(saved: dict, circuit: CallCircuit):
    for name in circuit.plastic:
        key = lambda s: zip(s[f"{name}.reader"].tolist(), s[f"{name}.pre"].tolist(), s[f"{name}.post"].tolist())
        w = dict(zip(key(saved), saved[f"{name}.w"].tolist()))
        circuit.sim.w[circuit.plastic[name].edges] = [w[k] for k in key(synapses(circuit))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=["cpu", "llm"], choices=["cpu", "llm"])
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs/learn"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    signs = np.load(SIGNS)
    kinds = [str(k) for k in signs["kinds"]]
    F = features(signs["frames"].astype(np.float32), signs["empty"].astype(np.float32))
    y, render = signs["label"].astype(int), signs["render"]
    group = np.array([GROUP_OF[kinds[v]] for v in y])
    train, test = np.nonzero(render != 2)[0], np.nonzero(render == 2)[0]
    reader = train_reader(F[train], y[train], len(kinds))
    accuracy = float(((F[test] @ reader).argmax(1) == y[test]).mean())

    fly = Connectome.load()
    visual, kc = mushroom_body(fly)
    fpu = compile_fpu(Netlist.fpu())
    fpu.load(build("reward.c"))
    ts = tools(fly, kinds, args.tools)
    brain = (len(fly), fly.pre, fly.post, fly.count)
    circuit = CallCircuit(brain, visual, kc, ts, reader, fpu=fpu)
    tester = CallCircuit(brain, visual, kc, ts, reader)
    print(f"reader held-out accuracy {accuracy:.2f}; " + "; ".join(
        f"{t.name}: {len(t.call)} call cells, {len(t.dans)} DANs, {len(circuit.plastic[t.name].edges)} plastic synapses"
        for t in ts), flush=True)

    start = time.time()
    circuit.respond(0)
    for t in ts:
        circuit.reference[t.name] = float(circuit.probe(t, t.reads[0]))
    print(f"DAN spikes per rewarded call {circuit.reference}  {time.time() - start:.0f} s", flush=True)

    held = {g: test[group[test] == g][:args.held] for g in GROUPS}
    curve, log = [], []

    def checkpoint(trial):
        saved = synapses(circuit)
        np.savez(args.out / f"synapses{trial}.npz", **saved)
        transfer(saved, tester)
        rates = {t.name: {} for t in ts}
        for g, ks in held.items():
            for name, rate in tester.call_rates(F[ks]).items():
                rates[name][g] = rate
        curve.append({"trial": trial} | rates)
        print(curve[-1], f"{time.time() - start:.0f} s", flush=True)
        (args.out / "learn.json").write_text(json.dumps(dict(curve=curve, log=log, reader_accuracy=accuracy,
                                                             reference=circuit.reference)))

    checkpoint(0)
    rng = np.random.default_rng(args.seed)
    for trial in range(args.trials):
        g = str(rng.choice(list(GROUPS), p=list(GROUPS.values())))
        k = int(rng.choice(train[group[train] == g]))
        t0 = circuit.sim.t
        spikes, dans = circuit.trial(F[k], trial)
        r = circuit.reinforce(spikes, dans)
        calls = {t.name: circuit.calls(spikes, t) for t in ts}
        log.append(dict(trial=trial, group=g, frame=k, calls=calls, dans=dans, r=r, steps=circuit.sim.t - t0))
        print(f"trial {trial} {g:6s} calls {calls} dans {dans} r {r} {(circuit.sim.t - t0) / 1e4:.1f} s", flush=True)
        if (trial + 1) % args.every == 0:
            checkpoint(trial + 1)


if __name__ == "__main__":
    main()
