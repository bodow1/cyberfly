import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cyberfly.lif import KICK, PARAMS, Sim
from cyberfly.neurons import Circuit, Kind
from cyberfly.runtime import boot_kicks, period

S = 2
GROUND = (11, 12, 14)
PANEL = (17, 19, 23)
LINE = (40, 43, 49)
FG = (236, 232, 225)
FG2 = (150, 156, 164)
FG3 = (88, 94, 102)
ACCENT = (240, 83, 58)
BLUE = (110, 160, 220)

MONO, DISPLAY = sys.argv[1], sys.argv[2]
LOW = 130
OUT = Path(__file__).parent
LEARN = [Path(a) for a in sys.argv[3:]] or [OUT.parent / "runs" / "learn" / "learn.json"]


def font(path, size):
    return ImageFont.truetype(path, size * S)


def canvas(w, h):
    img = Image.new("RGB", (w * S, h * S), GROUND)
    return img, ImageDraw.Draw(img)


def save(img, name):
    img.resize((img.width // S, img.height // S), Image.LANCZOS).save(OUT / name)


def trace(c: Circuit, watch: list[int], steps: int, kicks=None):
    sim = Sim(*c.arrays())
    k = boot_kicks(c)
    if kicks is not None:
        k = tuple(np.concatenate(x) for x in zip(k, kicks))
    o = np.argsort(k[0], kind="stable")
    k = tuple(x[o] for x in k)
    v, spikes = [], []
    for t in range(steps):
        m = k[0] == t
        st, si = sim.run(1, (k[0][m], k[1][m], k[2][m]) if m.any() else None)
        spikes += list(zip(st.tolist(), si.tolist()))
        v.append(sim.voltage(watch))
    return np.array(v), spikes


def nor_figure():
    p = period()
    k, lag, cycle = 12, 1, 2
    t0 = 1 + (k - 2 + cycle * 20) * p
    t1 = 1 + (k + 2 + cycle * 20) * p
    cases = [(0, 0), (1, 0), (0, 1), (1, 1)]
    W, H = 1600, 460
    img, d = canvas(W, H)
    small, label = font(MONO, 20), font(MONO, 22)
    pw = (W - 80 - 3 * 30) / 4
    for n, bits in enumerate(cases):
        c = Circuit(20)
        ins = []
        for v in bits:
            ph = k - lag
            a, b = c.latches[c.latch(v if ph % 2 == 0 else 1 - v)]
            ins.append(a if ph % 2 == 0 else b)
        clock = c.clock(k - 1)
        g = c.nor(ins, k, [lag, lag])
        v, spikes = trace(c, [g], t1)
        x0, y0 = 40 + n * (pw + 30), 90
        ph, plot_h = pw, 230
        d.rectangle((x0 * S, y0 * S, (x0 + ph) * S, (y0 + plot_h) * S), fill=PANEL)
        X = lambda t: (x0 + (t - t0) / (t1 - t0) * ph) * S
        Y = lambda mv: (y0 + plot_h - (min(max(mv, -LOW), -40) + LOW) / (LOW - 40) * plot_h) * S
        d.line((x0 * S, Y(PARAMS["v_th"]), (x0 + ph) * S, Y(PARAMS["v_th"])), fill=LINE, width=S)
        fired = [t for t, i in spikes if i == g and t0 <= t < t1]
        pts = [(X(t), Y(v[t, 0])) for t in range(t0, t1)]
        d.line(pts, fill=FG, width=2 * S)
        for t in fired:
            d.line((X(t), Y(v[t - 1, 0]), X(t), Y(-30)), fill=ACCENT, width=2 * S)
        for t, i in spikes:
            if t0 <= t < t1 and (i == clock or i in ins):
                yy = y0 + plot_h + (8 if i == clock else 26)
                d.polygon([(X(t), yy * S), (X(t) - 6 * S, (yy + 12) * S), (X(t) + 6 * S, (yy + 12) * S)],
                          fill=FG2 if i == clock else BLUE)
        for mv in (-LOW + 10, -60, -45):
            d.text(((x0 + 6) * S, Y(mv) - 2 * S), f"{mv}", font=small, fill=FG3, anchor="lb")
        d.text((x0 * S, (y0 - 34) * S), f"A={bits[0]}  B={bits[1]}", font=label, fill=FG)
        d.text(((x0 + ph) * S, (y0 - 34) * S), "fires" if fired else "silent", font=label,
               fill=ACCENT if fired else FG3, anchor="ra")
    d.text((40 * S, 12 * S), "membrane voltage (mV) of one NOR neuron over four phases, for each input case", font=small, fill=FG2)
    y = 90 + 230 + 60
    d.polygon([(40 * S, y * S), (34 * S, (y + 12) * S), (46 * S, (y + 12) * S)], fill=FG2)
    d.text((54 * S, (y - 2) * S), "clock spike (+242)", font=small, fill=FG2)
    d.polygon([(400 * S, y * S), (394 * S, (y + 12) * S), (406 * S, (y + 12) * S)], fill=BLUE)
    d.text((414 * S, (y - 2) * S), "input spike (−242)", font=small, fill=FG2)
    d.line((760 * S, (y + 6) * S, 790 * S, (y + 6) * S), fill=LINE, width=S)
    d.text((800 * S, (y - 2) * S), "threshold −45 mV", font=small, fill=FG2)
    d.line((1080 * S, y * S, 1080 * S, (y + 14) * S), fill=ACCENT, width=2 * S)
    d.text((1090 * S, (y - 2) * S), "output spike", font=small, fill=FG2)
    save(img, "fig_nor.png")


def raster(d, rows, spikes, t0, t1, box, names, colors, small):
    x0, y0, w, h = box
    rh = h / len(rows)
    for r, n in enumerate(rows):
        y = y0 + r * rh
        d.text(((x0 - 10) * S, (y + rh / 2) * S), names[r], font=small, fill=FG2, anchor="rm")
        d.line((x0 * S, (y + rh / 2) * S, (x0 + w) * S, (y + rh / 2) * S), fill=LINE, width=S)
        for t, i in spikes:
            if i == n and t0 <= t < t1:
                x = x0 + (t - t0) / (t1 - t0) * w
                d.rectangle(((x - 1.5) * S, (y + rh * 0.2) * S, (x + 1.5) * S, (y + rh * 0.8) * S), fill=colors[r])


def phase_axis(d, t0, t1, box, p, small, every=1):
    x0, y0, w, h = box
    first = int(np.ceil((t0 - 1) / p))
    for ph in range(first, int((t1 - 1) / p) + 1):
        x = x0 + (1 + ph * p - t0) / (t1 - t0) * w
        d.line((x * S, (y0 + h) * S, x * S, (y0 + h + 6) * S), fill=FG3, width=S)
        if ph % every == 0:
            d.text((x * S, (y0 + h + 10) * S), str(ph), font=small, fill=FG3, anchor="ma")


def clock_figure():
    p = period()
    c = Circuit(12)
    t1 = 1 + 30 * p
    _, spikes = trace(c, [c.ring[0]], t1)
    W, H = 1600, 440
    img, d = canvas(W, H)
    small = font(MONO, 20)
    box = (110, 70, W - 160, 280)
    raster(d, c.ring, spikes, 0, t1, box, [f"R{k}" for k in range(12)], [FG] * 12, small)
    phase_axis(d, 0, t1, box, p, small, every=2)
    d.text((110 * S, 30 * S), "clock ring of 12 neurons: each fires one phase (49 steps, 4.9 ms) after the one before",
           font=small, fill=FG2)
    d.text(((W - 50) * S, (70 + 280 + 30) * S), "phase", font=small, fill=FG3, anchor="ra")
    save(img, "fig_clock.png")


def latch_figure():
    p = period()
    c = Circuit(20)
    li = c.latch(0)
    a, b = c.latches[li]
    s = c.new(Kind.FOLLOW, 0, "set")
    r = c.new(Kind.FOLLOW, 0, "reset")
    c.latch_set(li, s)
    c.latch_reset(li, r)
    t_set = 1 + 6 * p - 1 - 18
    t_reset = 1 + 17 * p - 1 - 18
    kicks = (np.array([t_set, t_reset]), np.array([s, r], np.int32), np.full(2, KICK, np.float32))
    t1 = 1 + 26 * p
    _, spikes = trace(c, [a], t1, kicks)
    W, H = 1600, 300
    img, d = canvas(W, H)
    small = font(MONO, 20)
    box = (110, 60, W - 160, 170)
    raster(d, [s, r, a, b], spikes, 0, t1, box, ["set", "reset", "A (1)", "B (0)"], [BLUE, BLUE, ACCENT, FG], small)
    phase_axis(d, 0, t1, box, p, small, every=2)
    d.text((110 * S, 24 * S), "one bit: A fires on even phases while the bit is 1, B on odd phases while it is 0",
           font=small, fill=FG2)
    save(img, "fig_latch.png")


def system_figure():
    W, H = 1600, 520
    img, d = canvas(W, H)
    small, name = font(MONO, 20), font(DISPLAY, 26)

    def node(x, y, w, h, title, sub, hot=False, fill=PANEL):
        d.rounded_rectangle((x * S, y * S, (x + w) * S, (y + h) * S), radius=8 * S, fill=fill,
                            outline=ACCENT if hot else LINE, width=2 * S if hot else S)
        d.text(((x + w / 2) * S, (y + h / 2 - 10) * S), title, font=name, fill=FG, anchor="mm")
        d.text(((x + w / 2) * S, (y + h / 2 + 14) * S), sub, font=small, fill=FG2, anchor="mm")

    def arrow(pts, label=None, hot=False, below=False, seg=0):
        col = ACCENT if hot else FG3
        q = [(x * S, y * S) for x, y in pts]
        d.line(q, fill=col, width=2 * S)
        (x0, y0), (x1, y1) = q[-2], q[-1]
        n = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1)
        ux, uy, a = (x1 - x0) / n, (y1 - y0) / n, 11 * S
        d.polygon([(x1, y1), (x1 - a * ux - a * uy / 2, y1 - a * uy + a * ux / 2),
                   (x1 - a * ux + a * uy / 2, y1 - a * uy - a * ux / 2)], fill=col)
        if label:
            mx, my = (q[seg][0] + q[seg + 1][0]) / 2, (q[seg][1] + q[seg + 1][1]) / 2
            d.text((mx, my + (14 if below else -14) * S), label, font=small, fill=ACCENT if hot else FG2,
                   anchor="mm")

    d.rounded_rectangle((260 * S, 40 * S, 1340 * S, 480 * S), radius=14 * S, outline=LINE, width=S)
    d.text((280 * S, 58 * S), "native connectome: 138,639 neurons", font=small, fill=FG3)
    node(600, 90, 400, 110, "FPU + GPU", "186,381 added LIF neurons", hot=True)
    node(300, 330, 260, 90, "LC10c · LC4", "visual neurons")
    node(1040, 330, 260, 90, "DNa02 · DNp01", "descending neurons")
    node(670, 330, 260, 90, "rest of the brain", "existing synapses", fill=GROUND)
    node(20, 330, 220, 90, "eyes", "2 × 721 facets")
    node(1380, 330, 180, 90, "legs", "FlyGym")
    arrow([(240, 375), (300, 375)])
    arrow([(560, 375), (670, 375)])
    arrow([(930, 375), (1040, 375)])
    arrow([(1300, 375), (1380, 375)], "turn")
    arrow([(700, 200), (700, 260), (430, 260), (430, 330)], "output bits", hot=True, seg=1)
    arrow([(1170, 330), (1170, 260), (900, 260), (900, 200)], "input bits", hot=True, seg=1)
    save(img, "fig_system.png")


def learning_figure():
    runs = [json.loads(path.read_text())["curve"] for path in LEARN]
    W, H = 1600, 520
    img, d = canvas(W, H)
    small, name = font(MONO, 20), font(DISPLAY, 26)
    shades = {"arrow": ACCENT, "letter": BLUE, "stone": FG2, "blank": (110, 116, 124), "none": FG3}
    panels = {"cpu": "MBON09 calls the FPU", "llm": "MBON27 calls the LLM"}
    for n, (tool, title) in enumerate(panels.items()):
        curves = [c for c in runs if tool in c[0]]
        trials = [c["trial"] for c in curves[0]]
        last = trials[-1] or 1
        x0, y0, w, h = 90 + n * 790, 110, 560, 320
        d.text((x0 * S, 36 * S), title, font=name, fill=FG)
        d.text((x0 * S, 72 * S), f"share of held-out frames with a call, {len(curves)} runs", font=small, fill=FG3)
        for v in (0, 0.5, 1):
            y = y0 + (1 - v) * h
            d.line((x0 * S, y * S, (x0 + w) * S, y * S), fill=LINE, width=S)
            d.text(((x0 - 12) * S, y * S), f"{v:.1f}", font=small, fill=FG3, anchor="rm")
        for t in trials:
            d.text(((x0 + t / last * w) * S, (y0 + h + 14) * S), str(t), font=small, fill=FG3, anchor="ma")
        d.text(((x0 + w) * S, (y0 + h + 44) * S), "trials", font=small, fill=FG3, anchor="ra")
        point = lambda t, v: ((x0 + t / last * w) * S, (y0 + (1 - v) * h) * S)
        labels = []
        for g, col in reversed(shades.items()):
            faint = tuple((c + 2 * b) // 3 for c, b in zip(col, GROUND))
            for c in curves:
                d.line([point(e["trial"], e[tool][g]) for e in c], fill=faint, width=S)
            mean = [float(np.mean([c[k][tool][g] for c in curves])) for k in range(len(trials))]
            pts = [point(t, v) for t, v in zip(trials, mean)]
            d.line(pts, fill=col, width=3 * S)
            for x, y in pts:
                d.ellipse((x - 4 * S, y - 4 * S, x + 4 * S, y + 4 * S), fill=col)
            labels.append([pts[-1][1], g, col])
        labels.sort()
        for k in range(1, len(labels)):
            labels[k][0] = max(labels[k][0], labels[k - 1][0] + 24 * S)
        labels[-1][0] = min(labels[-1][0], (y0 + h) * S)
        for k in range(len(labels) - 2, -1, -1):
            labels[k][0] = min(labels[k][0], labels[k + 1][0] - 24 * S)
        for y, g, col in labels:
            d.text(((x0 + w + 16) * S, y), g, font=small, fill=col, anchor="lm")
    save(img, "fig_learning.png")

if __name__ == "__main__":
    nor_figure()
    clock_figure()
    latch_figure()
    system_figure()
    if all(path.exists() for path in LEARN):
        learning_figure()
