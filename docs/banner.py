import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from cyberfly.connectome import Connectome
from cyberfly.firmware import build
from cyberfly.iss import ISS
from cyberfly.lif import KICK, Sim
from cyberfly.raycaster import Raycaster

S = 2
WIDTH, HEIGHT = 1600, 500
OUT_SIZE = (1280, 400)
GROUND = (11, 12, 14)
DEEP = (4, 5, 6)
GRID = (17, 19, 23)
LINE = (40, 43, 49)
OFF = (30, 33, 38)
FAR = (78, 86, 98)
NEAR = (214, 218, 224)
FG = (236, 232, 225)
FG2 = (150, 156, 164)
FG3 = (88, 94, 102)
ACCENT = (240, 83, 58)
ACCENT_DIM = (120, 46, 36)
FRAMES = 60
FOLLOW_STEPS = 98
STEPS_PER_FRAME = 300
BRAIN_BOX = (40, 150, 780, 270)
TALKS_TO = ["LC4", "LC10c-1", "LC10c-2", "DNa02", "DNp01"]

MONO, DISPLAY = sys.argv[1], sys.argv[2]


def font(path, size):
    return ImageFont.truetype(path, size * S)


def replay(program):
    gpu = Raycaster()
    iss = ISS(program, gpu=gpu)
    resets = []
    last = 0
    while len(resets) < 2:
        assert not iss.halted
        iss.step()
        n = bin(gpu.demon).count("1")
        if n and last and n < last - 4:
            resets.append(iss.steps)
        last = n or last
    iss, gpu = ISS(program, gpu=(g := Raycaster())), g
    iss.run(resets[0])
    span = (resets[1] - resets[0]) / FRAMES
    states = []
    for f in range(FRAMES):
        iss.run(resets[0] + round((f + 1) * span))
        states.append(dict(r=list(iss.r), mem=list(iss.mem), pc=iss.pc, io=iss.io_out[-1], cam=gpu.cam,
                           dist=list(gpu.dist), demon=gpu.demon))
    return states


def groups(fly: Connectome, seed: int = 0):
    rng = np.random.default_rng(seed)

    def split(cells, k):
        cells = list(rng.permutation(cells))
        return [cells[i::k] for i in range(k)]

    left = split(fly.cells(["LC10c-1", "LC10c-2"], "left"), 16)
    right = split(fly.cells(["LC10c-1", "LC10c-2"], "right"), 16)
    lc4 = split(fly.cells(["LC4"]), 32)
    return [(left[c] if c < 16 else right[c - 16]) + lc4[c] for c in range(32)]


def spikes(fly: Connectome, states):
    sim = Sim(len(fly), fly.pre, fly.post, fly.count)
    cols = groups(fly)

    def kicks(demon, t0, steps):
        cells = [n for c in range(32) if (demon >> c) & 1 for n in cols[c]]
        times = np.arange(t0 - t0 % FOLLOW_STEPS + FOLLOW_STEPS, t0 + steps, FOLLOW_STEPS)
        if not cells or not len(times):
            return None
        t = np.repeat(times, len(cells))
        i = np.tile(np.array(cells, np.int32), len(times))
        return t.astype(np.int64), i, np.full(len(t), KICK, np.float32)

    sim.run(3000, kicks(states[0]["demon"], 0, 3000))
    out = []
    for st in states:
        t0 = sim.t
        _, i = sim.run(STEPS_PER_FRAME, kicks(st["demon"], t0, STEPS_PER_FRAME))
        out.append(np.unique(i))
    return out


class BrainView:
    def __init__(self, fly: Connectome, box):
        a = fly.annotations.dropna(subset=["pos_x", "pos_y", "pos_z"])
        x, y, z = (a[c].to_numpy(float) for c in ("pos_x", "pos_y", "pos_z"))
        x0, x1 = np.percentile(x, [0.3, 99.7])
        y0, y1 = np.percentile(y, [0.3, 99.7])
        bx, by, bw, bh = (v * S for v in box)
        k = min(bw / (x1 - x0), bh / (y1 - y0))
        ox, oy = bx + (bw - (x1 - x0) * k) / 2, by + (bh - (y1 - y0) * k) / 2
        px, py = ox + (x - x0) * k, oy + (y - y0) * k
        self.pos = np.full((len(fly), 2), np.nan)
        self.pos[a["index"].to_numpy()] = np.stack([px, py], 1)
        W, H = WIDTH * S, HEIGHT * S
        ix, iy = np.clip(px.astype(int), 0, W - 1), np.clip(py.astype(int), 0, H - 1)
        acc = np.zeros((H, W))
        depth = np.zeros((H, W))
        np.add.at(acc, (iy, ix), 1)
        zn = 1 - (z - np.percentile(z, 2)) / (np.percentile(z, 98) - np.percentile(z, 2))
        np.add.at(depth, (iy, ix), np.clip(zn, 0, 1))
        acc, depth = smooth(acc), smooth(depth)
        near = np.where(acc > 1e-3, depth / np.maximum(acc, 1e-3), 0)
        dens = np.clip(np.log1p(acc * 3) / np.log1p(np.percentile(acc[acc > 0.05], 99.5) * 3), 0, 1) ** 0.85
        mask = Image.fromarray((acc > 0.08).astype(np.uint8) * 255)
        soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(26 * S)), np.float32) / 255
        body = np.asarray(mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3 * S)), np.float32) / 255
        rim = np.clip(1 - np.abs(body - 0.5) * 6, 0, 1) * (body > 0.05)
        self.ground = np.clip(soft * 1.6, 0, 1)[..., None]
        shade = (np.array(FAR) + (np.array(NEAR) - np.array(FAR)) * near[..., None] ** 0.8)
        self.color = shade
        self.alpha = np.clip(dens * (0.6 + 0.6 * near), 0, 1)[..., None]
        self.rim = (rim * 0.45)[..., None]
        lit = a.cell_type.isin(TALKS_TO).to_numpy()
        self.lit = np.stack([px[lit], py[lit]], 1)

    def paint(self, img: np.ndarray) -> np.ndarray:
        img = img * (1 - self.ground) + np.array(DEEP) * self.ground
        img = img * (1 - self.rim) + np.array(FG3) * self.rim
        return img * (1 - self.alpha) + self.color * self.alpha


def smooth(a: np.ndarray) -> np.ndarray:
    k = np.array([1, 4, 6, 4, 1], np.float64) / 16
    a = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, a)
    return np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, a)


def glow(points, size, radius):
    layer = Image.new("L", size, 0)
    d = ImageDraw.Draw(layer)
    for u, v in points:
        d.ellipse((u - radius, v - radius, u + radius, v + radius), fill=255)
    return np.asarray(layer.filter(ImageFilter.GaussianBlur(radius * 2.2)), np.float32)[..., None] / 255


def bits(d, x, y, rows, cols, on, pitch, gap=S):
    for r in range(rows):
        for c in range(cols):
            d.rectangle((x + c * pitch, y + r * pitch, x + (c + 1) * pitch - gap, y + (r + 1) * pitch - gap),
                        fill=ACCENT if on(r, c) else OFF)


def block(d, x, y, w, h, label, small):
    d.rectangle((x, y, x + w, y + h), outline=LINE, width=S)
    d.text((x, y - 18 * S), label, font=small, fill=FG3)


def main(out: Path):
    fly = Connectome.load()
    states = replay(build("shooter.c"))
    fired = spikes(fly, states)
    view = BrainView(fly, BRAIN_BOX)
    small, tag = font(MONO, 13), font(MONO, 17)
    title = font(DISPLAY, 66)

    base = Image.new("RGB", (WIDTH * S, HEIGHT * S), GROUND)
    d = ImageDraw.Draw(base)
    for gx in range(0, WIDTH * S, 40 * S):
        d.line((gx, 0, gx, HEIGHT * S), fill=GRID, width=1)
    for gy in range(0, HEIGHT * S, 40 * S):
        d.line((0, gy, WIDTH * S, gy), fill=GRID, width=1)
    canvas = view.paint(np.asarray(base, np.float32))
    for u, v in view.lit:
        canvas[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2] = ACCENT_DIM
    base = Image.fromarray(canvas.astype(np.uint8))
    d = ImageDraw.Draw(base)
    dx = 900 * S
    d.text((40 * S, 30 * S), "CYBERFLY", font=title, fill=FG)
    d.text((44 * S, 114 * S), "a 32-bit RISC-V computer compiled into the brain of a fruit fly", font=tag, fill=FG2)
    block(d, dx, 170 * S, 300 * S, 250 * S, "RAM  256 x 32", small)
    block(d, dx + 330 * S, 170 * S, 140 * S, 140 * S, "REGS  x1-x31", small)
    block(d, dx + 500 * S, 170 * S, 160 * S, 250 * S, "GPU  32 rays", small)
    block(d, dx + 330 * S, 340 * S, 140 * S, 80 * S, "PC  IO  CAM", small)
    d.line((dx - 20 * S, 170 * S, dx - 20 * S, 420 * S), fill=ACCENT, width=S)
    d.rectangle((44 * S, 436 * S, 50 * S, 442 * S), fill=ACCENT)
    d.text((58 * S, 431 * S), "LC4  LC10c  DNa02  DNp01: the fly neurons the FPU talks to", font=small, fill=FG3)
    stats = "138,639 fly neurons   ·   83,864-neuron FPU   ·   raycasting GPU   ·   0 mismatches"
    d.text((44 * S, 462 * S), stats, font=small, fill=FG3)

    frames = []
    hub = (np.nanmean(view.lit[:, 0]), np.nanmean(view.lit[:, 1]))
    for f, (st, ids) in enumerate(zip(states, fired)):
        img = base.copy()
        d = ImageDraw.Draw(img)
        bits(d, dx + 6 * S, 223 * S, 64, 128, lambda r, c: (st["mem"][r * 4 + c // 32] >> (31 - c % 32)) & 1, 2.25 * S)
        bits(d, dx + 336 * S, 178 * S, 31, 32, lambda r, c: (st["r"][r + 1] >> (31 - c)) & 1, 4 * S)
        for k, word in enumerate([st["pc"], st["io"], st["cam"]]):
            bits(d, dx + 336 * S, (352 + 20 * k) * S, 1, 32, lambda r, c, w=word: (w >> (31 - c)) & 1, 4 * S)
        for c, q in enumerate(st["dist"]):
            h = (1 - q / 31) * 220 * S + 6 * S
            x = dx + 508 * S + c * 4.6 * S
            hot = (st["demon"] >> c) & 1
            d.rectangle((x, 295 * S - h / 2, x + 3 * S, 295 * S + h / 2), fill=ACCENT if hot else FG3)
        phase = (f % 12) / 12
        x0, y0, x1, y1 = hub[0], hub[1], dx - 20 * S, 295 * S
        d.line((x0, y0, x1, y1), fill=ACCENT_DIM, width=S)
        for j in range(3):
            p = (phase + j / 3) % 1
            u, v = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
            d.ellipse((u - 3 * S, v - 3 * S, u + 3 * S, v + 3 * S), fill=ACCENT)
        pts = view.pos[ids]
        pts = pts[~np.isnan(pts[:, 0])]
        arr = np.asarray(img, np.float32)
        halo = glow(pts, img.size, 3 * S)
        arr = arr * (1 - halo * 0.8) + np.array(ACCENT) * halo * 0.8
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for u, v in pts:
            d.ellipse((u - 1.6 * S, v - 1.6 * S, u + 1.6 * S, v + 1.6 * S), fill=(255, 150, 120))
        frames.append(img.resize(OUT_SIZE, Image.LANCZOS))
    sample = Image.new("RGB", (OUT_SIZE[0], OUT_SIZE[1] * 2))
    sample.paste(frames[0], (0, 0))
    sample.paste(frames[FRAMES // 2], (0, OUT_SIZE[1]))
    palette = sample.quantize(255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=False)
    print(len(frames), "frames", [len(i) for i in fired][:10])


if __name__ == "__main__":
    main(Path(__file__).with_name("banner.gif"))
