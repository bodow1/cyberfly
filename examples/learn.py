import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from cyberfly.connectome import Connectome
from cyberfly.learning import CallCircuit, features, train_reader

SIGNS = Path(__file__).resolve().parent.parent / "data" / "signs.npz"
ARROWS = {"left", "right"}
GROUPS = {"arrow": 0.4, "stone": 0.25, "blank": 0.15, "none": 0.2}
VISUAL = ("visual_projection", "optic", "visual_centrifugal")
BG, LINE, FG, DIM, ACCENT = (11, 12, 14), (40, 43, 49), (236, 232, 225), (88, 94, 102), (240, 83, 58)
SHADES = {"arrow": ACCENT, "stone": (150, 156, 164), "blank": (110, 116, 124), "none": (70, 75, 82)}


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


def plot(curve: list[dict], path: Path):
    W, H, S = 900, 420, 2
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(13 * S)
    x0, y0, x1, y1 = 70 * S, 40 * S, (W - 150) * S, (H - 60) * S
    for v in (0, 0.5, 1):
        y = y1 - v * (y1 - y0)
        d.line((x0, y, x1, y), fill=LINE, width=S)
        d.text((x0 - 12 * S, y), f"{v:.1f}", font=font, fill=DIM, anchor="rm")
    last = curve[-1]["trial"] or 1
    for g, col in SHADES.items():
        pts = [(x0 + c["trial"] / last * (x1 - x0), y1 - c[g] * (y1 - y0)) for c in curve]
        d.line(pts, fill=col, width=3 * S)
        for p in pts:
            d.ellipse((p[0] - 4 * S, p[1] - 4 * S, p[0] + 4 * S, p[1] + 4 * S), fill=col)
        d.text((x1 + 14 * S, pts[-1][1]), g, font=font, fill=col, anchor="lm")
    for c in curve:
        x = x0 + c["trial"] / last * (x1 - x0)
        d.text((x, y1 + 10 * S), str(c["trial"]), font=font, fill=DIM, anchor="ma")
    d.text((x1, y1 + 30 * S), "trials", font=font, fill=DIM, anchor="ra")
    d.text((x0, 14 * S), "MBON09 call rate on held-out sign frames", font=font, fill=FG)
    img.resize((W, H), Image.LANCZOS).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=240)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/learn"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    signs = np.load(SIGNS)
    kinds = [str(k) for k in signs["kinds"]]
    F = features(signs["frames"].astype(np.float32), signs["empty"].astype(np.float32))
    y, render = signs["label"].astype(int), signs["render"]
    group = np.array(["arrow" if kinds[v] in ARROWS else kinds[v] for v in y])
    train, test = np.nonzero(render != 2)[0], np.nonzero(render == 2)[0]
    reader = train_reader(F[train], y[train], len(kinds))
    accuracy = float(((F[test] @ reader).argmax(1) == y[test]).mean())

    fly = Connectome.load()
    visual, kc = mushroom_body(fly)
    call = fly.cells(["MBON09"])
    circuit = CallCircuit((len(fly), fly.pre, fly.post, fly.count), visual, kc, call, reader)
    print(f"reader held-out accuracy {accuracy:.2f}; {len(call)} MBON09 cells; "
          f"{len(circuit.plastic.edges)} plastic synapses", flush=True)

    held = {g: test[group[test] == g][:10] for g in GROUPS}

    def evaluate(trial):
        return {"trial": trial} | {g: circuit.call_rate(F[ks]) for g, ks in held.items()}

    rng = np.random.default_rng(3)
    curve, log, start = [evaluate(0)], [], time.time()
    print(curve[-1], flush=True)
    for t in range(args.trials):
        g = str(rng.choice(list(GROUPS), p=list(GROUPS.values())))
        k = int(rng.choice(train[group[train] == g]))
        spikes = circuit.look(F[k], t, explore_hz=2.0)
        pays = rng.random() < 0.3 if args.shuffle else g == "arrow"
        r = circuit.reinforce(spikes, float(pays))
        log.append(dict(trial=t, group=g, calls=circuit.calls(spikes), r=r))
        if (t + 1) % args.every == 0:
            curve.append(evaluate(t + 1))
            print(curve[-1], f"{time.time() - start:.0f} s", flush=True)
    (args.out / "learn.json").write_text(json.dumps(dict(curve=curve, log=log, reader_accuracy=accuracy)))
    plot(curve, args.out / "call_rate.png")


if __name__ == "__main__":
    main()
