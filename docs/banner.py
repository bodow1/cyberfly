import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cyberfly.connectome import Connectome
from cyberfly.firmware import build
from cyberfly.iss import ISS
from cyberfly.raycaster import Raycaster, render

S = 2
WIDTH, HEIGHT = 1600 * S, 500 * S
GROUND = (9, 11, 12)
GRID = (17, 21, 23)
DIM = (40, 47, 51)
CLOUD = (64, 74, 80)
MID = (92, 104, 110)
INK = (232, 236, 238)
ACCENT = (232, 86, 70)

MONO = sys.argv[1] if len(sys.argv) > 1 else "DejaVuSansMono.ttf"
DISPLAY = sys.argv[2] if len(sys.argv) > 2 else "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"


def font(path, size):
    return ImageFont.truetype(path, size * S)


def brain(draw: ImageDraw.ImageDraw, fly: Connectome, box):
    a = fly.annotations
    x, y = a.pos_x.to_numpy(float), a.pos_y.to_numpy(float)
    x0, x1 = np.percentile(x, [0.5, 99.5])
    y0, y1 = np.percentile(y, [0.5, 99.5])
    bx, by, bw, bh = box
    scale = min(bw / (x1 - x0), bh / (y1 - y0))
    ox = bx + (bw - (x1 - x0) * scale) / 2
    oy = by + (bh - (y1 - y0) * scale) / 2
    px, py = ox + (x - x0) * scale, oy + (y - y0) * scale
    for u, v in zip(px, py):
        draw.point((u, v), fill=CLOUD)
    lit = a.cell_type.isin(["LC4", "LC10c-1", "LC10c-2", "DNa02", "DNp01"]).to_numpy()
    for u, v in zip(px[lit], py[lit]):
        draw.rectangle((u - S, v - S, u + S, v + S), fill=ACCENT)
    return ox + (x1 - x0) * scale / 2, oy + (y1 - y0) * scale / 2


def bits(draw, x, y, rows, cols, on, pitch, gap=S):
    for r in range(rows):
        for c in range(cols):
            fill = ACCENT if on(r, c) else DIM
            draw.rectangle((x + c * pitch, y + r * pitch, x + (c + 1) * pitch - gap, y + (r + 1) * pitch - gap),
                           fill=fill)


def block(draw, x, y, w, h, label, small):
    draw.rectangle((x, y, x + w, y + h), outline=MID, width=S)
    draw.text((x, y - 18 * S), label, font=small, fill=MID)


def main(out: Path):
    img = Image.new("RGB", (WIDTH, HEIGHT), GROUND)
    d = ImageDraw.Draw(img)
    for gx in range(0, WIDTH, 40 * S):
        d.line((gx, 0, gx, HEIGHT), fill=GRID, width=1)
    for gy in range(0, HEIGHT, 40 * S):
        d.line((0, gy, WIDTH, gy), fill=GRID, width=1)

    small = font(MONO, 13)
    tag = font(MONO, 17)
    fly = Connectome.load()
    cx, cy = brain(d, fly, (40 * S, 150 * S, 760 * S, 270 * S))

    program = build("shooter.c")
    gpu = Raycaster()
    iss = ISS(program, gpu=gpu).run(600)
    ram = iss.mem
    dx = 900 * S
    block(d, dx, 170 * S, 300 * S, 250 * S, "RAM  256 x 32", small)
    bits(d, dx + 6 * S, 223 * S, 64, 128, lambda r, c: (ram[r * 4 + c // 32] >> (31 - c % 32)) & 1, 2.25 * S)
    block(d, dx + 330 * S, 170 * S, 140 * S, 140 * S, "REGS  x1-x31", small)
    bits(d, dx + 336 * S, 178 * S, 31, 32, lambda r, c: (iss.r[r + 1] >> (31 - c)) & 1, 4 * S)
    block(d, dx + 500 * S, 170 * S, 160 * S, 250 * S, "GPU  32 rays", small)
    rays = render(0x780, 0x880, 0)
    for c, q in enumerate(rays):
        h = int((1 - q / 31) * 220 * S) + 6 * S
        mid = 295 * S
        d.rectangle((dx + 508 * S + c * 4.6 * S, mid - h / 2, dx + 508 * S + c * 4.6 * S + 3 * S, mid + h / 2),
                    fill=ACCENT if q < 12 else MID)
    block(d, dx + 330 * S, 340 * S, 140 * S, 80 * S, "PC  IO  CAM", small)
    for k, word in enumerate([iss.pc, iss.io_out[-1], gpu.cam]):
        bits(d, dx + 336 * S, (352 + 20 * k) * S, 1, 32, lambda r, c, w=word: (w >> (31 - c)) & 1, 4 * S)

    d.line((cx, cy, dx - 20 * S, 295 * S), fill=ACCENT, width=S)
    d.rectangle((cx - 4 * S, cy - 4 * S, cx + 4 * S, cy + 4 * S), outline=ACCENT, width=S)
    d.line((dx - 20 * S, 170 * S, dx - 20 * S, 420 * S), fill=ACCENT, width=S)

    d.rectangle((44 * S, 434 * S, 50 * S, 440 * S), fill=ACCENT)
    d.text((58 * S, 430 * S), "LC4  LC10c  DNa02  DNp01: the fly neurons the FPU talks to", font=small, fill=MID)
    d.text((40 * S, 36 * S), "CYBERFLY", font=font(DISPLAY, 64), fill=INK)
    d.text((44 * S, 116 * S), "a 32-bit RISC-V computer compiled into the brain of a fruit fly", font=tag, fill=MID)
    stats = "138,639 fly neurons   ·   83,864-neuron FPU   ·   raycasting GPU   ·   0 mismatches"
    d.text((44 * S, 462 * S), stats, font=small, fill=MID)
    img.resize((WIDTH // S, HEIGHT // S), Image.LANCZOS).save(out)


if __name__ == "__main__":
    main(Path(__file__).with_name("banner.png"))
