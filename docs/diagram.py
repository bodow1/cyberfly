import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
WIDTH, HEIGHT = 1600, 820
GROUND = (11, 12, 14)
PANEL = (17, 19, 23)
LINE = (40, 43, 49)
FG = (236, 232, 225)
FG2 = (150, 156, 164)
FG3 = (88, 94, 102)
ACCENT = (240, 83, 58)

MONO, DISPLAY = sys.argv[1], sys.argv[2]


def font(path, size):
    return ImageFont.truetype(path, size * S)


def box(d, x, y, w, h, title, sub=None, hot=False, fill=PANEL):
    d.rounded_rectangle((x * S, y * S, (x + w) * S, (y + h) * S), radius=8 * S, fill=fill,
                        outline=ACCENT if hot else LINE, width=2 * S if hot else S)
    cy = y + h / 2 - (10 if sub else 0)
    d.text(((x + w / 2) * S, cy * S), title, font=font(DISPLAY, 19), fill=FG, anchor="mm")
    if sub:
        d.text(((x + w / 2) * S, (cy + 24) * S), sub, font=font(MONO, 13), fill=FG2, anchor="mm")
    return (x, y, w, h)


def arrow(d, pts, label=None, hot=False, at=0.5, offset=(0, -14), seg=0):
    col = ACCENT if hot else FG3
    pts = [(x * S, y * S) for x, y in pts]
    d.line(pts, fill=col, width=2 * S, joint="curve")
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    ux, uy = (x1 - x0), (y1 - y0)
    n = max((ux * ux + uy * uy) ** 0.5, 1)
    ux, uy = ux / n, uy / n
    a = 11 * S
    d.polygon([(x1, y1), (x1 - a * ux - a * 0.5 * uy, y1 - a * uy + a * 0.5 * ux),
               (x1 - a * ux + a * 0.5 * uy, y1 - a * uy - a * 0.5 * ux)], fill=col)
    if label:
        (sx, sy), (ex, ey) = pts[seg], pts[seg + 1]
        mx, my = sx + (ex - sx) * at, sy + (ey - sy) * at
        d.text((mx + offset[0] * S, my + offset[1] * S), label, font=font(MONO, 13), fill=FG2 if not hot else ACCENT,
               anchor="mm")


def main(out: Path):
    img = Image.new("RGB", (WIDTH * S, HEIGHT * S), GROUND)
    d = ImageDraw.Draw(img)
    kick = font(MONO, 13)

    d.text((40 * S, 34 * S), "BUILD", font=kick, fill=FG3)
    box(d, 40, 60, 200, 64, "C program", "firmware/*.c")
    box(d, 330, 60, 200, 64, "RV32I binary", "words in RAM")
    arrow(d, [(240, 92), (330, 92)], "clang")
    box(d, 40, 160, 200, 64, "Verilog", "FPU + GPU")
    box(d, 330, 160, 200, 64, "Gate netlist", "NOR · NOT · DFF")
    box(d, 620, 160, 240, 64, "LIF neurons", "clock ring · gates · latches")
    arrow(d, [(240, 192), (330, 192)], "yosys")
    arrow(d, [(530, 192), (620, 192)], "compiler")
    arrow(d, [(530, 92), (740, 92), (740, 160)], "loaded into RAM latches", at=0.45)

    d.rounded_rectangle((300 * S, 290 * S, 1300 * S, 770 * S), radius=14 * S, outline=LINE, width=S)
    d.text((1280 * S, 312 * S), "FLYWIRE CONNECTOME  ·  138,639 NEURONS  ·  SHIU LIF MODEL", font=kick, fill=FG3, anchor="rm")
    arrow(d, [(740, 224), (740, 340)])

    box(d, 600, 340, 280, 110, "FPU + GPU", "83,864 to 186,381 neurons", hot=True)
    box(d, 360, 560, 220, 80, "LC10c · LC4", "visual projection")
    box(d, 1020, 560, 240, 80, "DNa02 · DNp01", "descending")
    box(d, 720, 560, 200, 80, "central brain", "native synapses", fill=GROUND)
    box(d, 720, 680, 200, 64, "MBON09", "mushroom body output")

    arrow(d, [(640, 450), (640, 500), (470, 500), (470, 560)], "io_out · followers", hot=True, at=0.5, offset=(0, -14), seg=1)
    arrow(d, [(580, 600), (720, 600)])
    arrow(d, [(920, 600), (1020, 600)])
    arrow(d, [(1140, 560), (1140, 440), (880, 440)], "capture neurons · io_in", hot=True, at=0.5, offset=(0, 14), seg=1)
    arrow(d, [(920, 712), (980, 712), (980, 500), (840, 500), (840, 450)])
    d.text((972 * S, 660 * S), "learned call · io_in", font=kick, fill=FG2, anchor="rm")
    arrow(d, [(580, 620), (650, 620), (650, 712), (720, 712)])
    d.text((640 * S, 742 * S), "via Kenyon cells", font=kick, fill=FG3, anchor="mm")


    box(d, 40, 560, 200, 80, "Compound eyes", "2 × 721 ommatidia")
    arrow(d, [(240, 600), (360, 600)])
    box(d, 1360, 560, 200, 80, "Legs", "FlyGym · MuJoCo")
    arrow(d, [(1260, 600), (1360, 600)], "turn · stop", offset=(0, -14))

    box(d, 1360, 340, 200, 80, "Radio", "1 TX · 1 RX neuron")
    arrow(d, [(1360, 362), (880, 362)], "program frames", at=0.5, offset=(0, -12))
    arrow(d, [(880, 398), (1360, 398)], "request", at=0.5, offset=(0, 14))

    img.resize((WIDTH, HEIGHT), Image.LANCZOS).save(out)


if __name__ == "__main__":
    main(Path(__file__).with_name("diagram.png"))
