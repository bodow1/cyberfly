import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from cyberfly.compiler import compile_fpu
from cyberfly.connectome import Connectome
from cyberfly.firmware import build
from cyberfly.hdl import Netlist
from cyberfly.learning import CallCircuit, Tool, features, train_reader

SIGNS = Path(__file__).resolve().parent.parent / "data" / "signs.npz"
GROUP_OF = {"left": "arrow", "right": "arrow", "letter": "letter", "stone": "stone", "blank": "blank", "none": "none"}
GROUPS = {"arrow": 0.3, "letter": 0.3, "stone": 0.15, "blank": 0.1, "none": 0.15}
PAYS = {"cpu": "arrow", "llm": "letter"}
TITLES = {"cpu": "MBON09 calls the FPU", "llm": "MBON27 calls the LLM"}
VISUAL = ("visual_projection", "optic", "visual_centrifugal")
BG, LINE, FG, DIM, ACCENT = (11, 12, 14), (40, 43, 49), (236, 232, 225), (88, 94, 102), (240, 83, 58)
SHADES = {"arrow": ACCENT, "letter": (110, 160, 220), "stone": (150, 156, 164), "blank": (110, 116, 124),
          "none": (70, 75, 82)}


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


def tools(fly: Connectome) -> list[Tool]:
    cells = lambda types: np.array(fly.cells(types))
    return [
        Tool("cpu", cells(["MBON09"]), cells(["PAM12", "PAM13", "PAM14"]), 3, 12, 30, 0.2),
        Tool("llm", cells(["MBON27"]), cells(["PAM01", "PAM08"]), 10, 13, 29, 0.3),
    ]


def plot(curve: list[dict], path: Path):
    W, H, S = 1400, 440, 2
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(14 * S)
    last = curve[-1]["trial"] or 1
    for n, tool in enumerate(TITLES):
        x0, y0 = (70 + n * 690) * S, 50 * S
        x1, y1 = x0 + 500 * S, (H - 60) * S
        d.text((x0, 16 * S), TITLES[tool], font=font, fill=FG)
        for v in (0, 0.5, 1):
            y = y1 - v * (y1 - y0)
            d.line((x0, y, x1, y), fill=LINE, width=S)
            d.text((x0 - 12 * S, y), f"{v:.1f}", font=font, fill=DIM, anchor="rm")
        labels = []
        for g, col in SHADES.items():
            pts = [(x0 + c["trial"] / last * (x1 - x0), y1 - c[tool][g] * (y1 - y0)) for c in curve]
            d.line(pts, fill=col, width=3 * S)
            for p in pts:
                d.ellipse((p[0] - 4 * S, p[1] - 4 * S, p[0] + 4 * S, p[1] + 4 * S), fill=col)
            labels.append([pts[-1][1], g, col])
        labels.sort()
        for k in range(1, len(labels)):
            labels[k][0] = max(labels[k][0], labels[k - 1][0] + 18 * S)
        for y, g, col in labels:
            d.text((x1 + 14 * S, y), g, font=font, fill=col, anchor="lm")
        for c in curve:
            d.text((x0 + c["trial"] / last * (x1 - x0), y1 + 10 * S), str(c["trial"]), font=font, fill=DIM,
                   anchor="ma")
        d.text((x1, y1 + 32 * S), "trials", font=font, fill=DIM, anchor="ra")
    img.resize((W, H), Image.LANCZOS).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--held", type=int, default=8)
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
    ts = tools(fly)
    circuit = CallCircuit((len(fly), fly.pre, fly.post, fly.count), visual, kc, ts, reader, fpu=fpu)
    print(f"reader held-out accuracy {accuracy:.2f}; " + "; ".join(
        f"{t.name}: {len(t.call)} call cells, {len(t.dans)} DANs, {len(circuit.plastic[t.name].edges)} plastic synapses"
        for t in ts), flush=True)

    start = time.time()
    rng = np.random.default_rng(11)
    for t in ts:
        counts = []
        for k in rng.choice(train[group[train] == PAYS[t.name]], 4, replace=False):
            spikes = circuit.look(F[k], 900_000 + int(k))
            called = {u.name for u in ts if circuit.calls(spikes, u)}
            dans = circuit.respond(F[k], 900_000 + int(k), called)
            if t.name in called:
                counts.append(dans[t.name])
        circuit.reference[t.name] = float(np.median(counts)) if counts else 1.0
    print(f"reference DAN spikes per rewarded call {circuit.reference}  {time.time() - start:.0f} s", flush=True)

    held = {g: test[group[test] == g][:args.held] for g in GROUPS}

    def evaluate(trial):
        rates = {t.name: {} for t in ts}
        for g, ks in held.items():
            for name, rate in circuit.call_rates(F[ks]).items():
                rates[name][g] = rate
        return {"trial": trial} | rates

    curve, log = [evaluate(0)], []
    print(curve[-1], f"{time.time() - start:.0f} s", flush=True)
    rng = np.random.default_rng(3)
    for trial in range(args.trials):
        g = str(rng.choice(list(GROUPS), p=list(GROUPS.values())))
        k = int(rng.choice(train[group[train] == g]))
        spikes = circuit.look(F[k], trial, explore_hz=2.0)
        called = {t.name for t in ts if circuit.calls(spikes, t)}
        dans = circuit.respond(F[k], trial, called) if called else {t.name: 0 for t in ts}
        r = circuit.reinforce(spikes, dans)
        log.append(dict(trial=trial, group=g, called=sorted(called), dans=dans, r=r, t=circuit.sim.t))
        print(f"trial {trial} {g:6s} called {sorted(called)} dans {dans} r {r}", flush=True)
        if (trial + 1) % args.every == 0:
            curve.append(evaluate(trial + 1))
            print(curve[-1], f"{time.time() - start:.0f} s", flush=True)
            (args.out / "learn.json").write_text(json.dumps(dict(curve=curve, log=log, reader_accuracy=accuracy,
                                                                 reference=circuit.reference)))
    plot(curve, args.out / "call_rate.png")


if __name__ == "__main__":
    main()
