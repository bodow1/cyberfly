"""Draw the compilation figure and bootloader loop with the original figure styles.

Run with the same font arguments as figures.py (JetBrains Mono Regular, Archivo Bold).
The bootloader states come from boot_trace.json, generated with boot_trace.py.
"""

import json
from bisect import bisect_right
from dataclasses import dataclass

from PIL import Image

from cyberfly.radio import BITS, SLOT, frame_spikes, program_frames
from figures import ACCENT, BLUE, DISPLAY, FG, FG2, FG3, GROUND, LINE, MONO, OUT, PANEL, S, canvas, font, save


def text(d, x, y, value, face, color=FG, anchor="lm"):
    d.text((x * S, y * S), value, font=face, fill=color, anchor=anchor)


def line(d, points, color=LINE, width=1):
    d.line([(x * S, y * S) for x, y in points], fill=color, width=width * S)


def arrow(d, x1, x2, y, color):
    direction = 1 if x2 > x1 else -1
    line(d, [(x1, y), (x2, y)], color, 2)
    d.polygon([(x2 * S, y * S), ((x2 - direction * 10) * S, (y - 5) * S),
               ((x2 - direction * 10) * S, (y + 5) * S)], fill=color)


def enclosure(d, box):
    d.rounded_rectangle(tuple(v * S for v in box), radius=12 * S, fill=PANEL, outline=LINE, width=S)


def squares(d, values, x, y, pitch, height, on=ACCENT, off=LINE, group_gap=0):
    for i, bit in enumerate(values):
        left = x + i * pitch + (i // 4) * group_gap
        d.rectangle((left * S, y * S, (left + pitch - 2) * S, (y + height) * S),
                    fill=on if bit else off)


def bits(word, width):
    return [(word >> bit) & 1 for bit in range(width - 1, -1, -1)]


def compilation(trace):
    img, d = canvas(1600, 430)
    label, small, title = font(MONO, 20), font(MONO, 17), font(DISPLAY, 27)
    enclosure(d, (924, 18, 1570, 411))
    text(d, 950, 48, "Compiled FPU", title)
    text(d, 1543, 49, "83,864 neurons", small, FG2, "rm")
    line(d, [(950, 75), (1544, 75)])

    # Two parallel paths meet inside one computer.
    for y, source, detail, tool, result, detail2, verb in (
        (142, "CPU design", "hw/fpu.v", "Yosys", "Gate netlist", "NOR, NOT, flip-flops", "schedule"),
        (306, "Boot program", "boot.c + crt0.S", "clang", "RISC-V words", trace["boot_first_word"], "load"),
    ):
        text(d, 40, y - 15, source, title)
        text(d, 40, y + 22, detail, label, FG2)
        arrow(d, 275, 398, y, FG3)
        text(d, 336, y - 20, tool, small, FG2, "mm")
        text(d, 430, y - 15, result, title)
        text(d, 430, y + 22, detail2, label, FG2)
        arrow(d, 740, 925, y, ACCENT)
        text(d, 832, y - 20, verb, small, ACCENT, "mm")
    text(d, 40, 198, "regs[7][0]", small, ACCENT)
    text(d, 430, 198, "x7[0] flip-flop", small, ACCENT)
    text(d, 430, 361, "addi sp, x0, 1024", small, FG2)

    text(d, 950, 109, "Register x7", title)
    text(d, 1543, 110, "32 latch pairs", small, FG2, "rm")
    text(d, 970, 149, "A", small, FG2)
    text(d, 970, 172, "B", small, FG2)
    start, pitch = 1006, 16.6
    for i in range(32):
        x = start + i * pitch
        line(d, [(x, 150), (x, 171)], FG3)
        for y in (149, 172):
            d.ellipse(((x - 4.4) * S, (y - 4.4) * S, (x + 4.4) * S, (y + 4.4) * S),
                      fill=PANEL, outline=FG3, width=2 * S)
    bit0 = start + 31 * pitch
    d.rounded_rectangle(((bit0 - 7) * S, 140 * S, (bit0 + 7) * S, 181 * S),
                        radius=4 * S, outline=ACCENT, width=S)
    text(d, start, 201, "31", small, FG2, "mm")
    text(d, start + 31 * pitch, 201, "bit 0", small, ACCENT, "mm")
    line(d, [(950, 228), (1544, 228)])

    text(d, 950, 267, "RAM word 0", title)
    text(d, 1543, 268, "address 0x000", small, FG2, "rm")
    word = int(trace["boot_first_word"], 16)
    assert word == 0x40000113  # addi sp, x0, 1024
    squares(d, bits(word, 32), 990, 298, 16.4, 28, group_gap=1.2)
    text(d, 990, 351, "bit 31", small, FG2)
    text(d, 1523, 351, "bit 0", small, FG2, "rm")
    save(img, "fig_compilation.png")


@dataclass
class Beat:
    title: str
    sent: bool = False
    tx: int | None = None
    rx: int | None = None
    slots: int = 0
    writes: int = 0
    packet: str = ""
    pc: str = "bootloader"
    executed: bool = False


class Upload:
    def __init__(self, trace):
        self.trace = trace
        self.words = [int(w["word"], 16) for w in trace["writes"]]
        self.cycles = [w["cycle"] for w in trace["writes"]]
        self.frames = program_frames(self.words)

    def receive(self, title, index, progress, writes, packet):
        slots = min(BITS + 1, int(progress * (BITS + 4)))
        return Beat(title, sent=True, rx=self.frames[index], slots=slots, writes=writes, packet=packet)

    def at(self, t):
        if t < 1.1:
            slots = min(BITS + 1, int(t * 19))
            return Beat("The bootloader sends a request", tx=ord("?"), sent=slots == BITS + 1, slots=slots)
        if t < 1.85:
            return self.receive("The host sends the program length", 0, (t - 1.1) / .75, 0, "header")
        if t < 4.1:
            progress = (t - 1.85) / .75
            part = int(progress)
            return self.receive("Three frames carry the first instruction", 1 + part, progress - part,
                                0, f"word 0   frame {part + 1}/3")
        if t < 4.62:
            return Beat("The first word lands at 0x200", sent=True, writes=1, packet=self.trace["writes"][0]["word"])
        if t < 5.85:
            progress = (t - 4.62) / 1.23
            cycle = self.cycles[0] + progress * (self.cycles[-1] - self.cycles[0])
            written = bisect_right(self.cycles, cycle)
            return Beat("The bootloader writes each word to RAM", sent=True, writes=written,
                        packet=f"{written}/{len(self.words)} words written")
        if t < 6.6:
            return self.receive("The bootloader checks the checksum", -1, (t - 5.85) / .75,
                                len(self.words), "checksum")
        if t < 7.24:
            return Beat("The program counter jumps to the uploaded code", sent=True, writes=len(self.words),
                        packet="checksum matched", pc=self.trace["entry_address"])
        return Beat("The FPU executes the first instruction", sent=True, writes=len(self.words),
                    packet="checksum matched", pc=self.trace["pc_after_entry_instruction"], executed=True)


def raster(d, x, y, word, slots, color):
    pitch = 25
    line(d, [(x - 8, y), (x + BITS * pitch + 9, y)])
    for j in range(BITS + 1):
        line(d, [(x + j * pitch, y + 6), (x + j * pitch, y + 10)], LINE)
    if word is None:
        return
    for t in frame_spikes(word, 0, 1):
        slot = int(t) // SLOT
        if slot < slots:
            xx = x + slot * pitch
            line(d, [(xx, y - 16), (xx, y + 2)], color, 3)


def render(upload, beat, fonts):
    small, label, title = fonts
    img, d = canvas(1600, 460)
    text(d, 40, 31, beat.title, label, FG2)
    enclosure(d, (350, 62, 1568, 431))
    text(d, 40, 108, "Host", title)
    text(d, 377, 108, "Fly Processing Unit", title)
    line(d, [(937, 134), (937, 406)])

    text(d, 40, 169, "Program image" if beat.sent else "Waiting for '?'", label, FG if beat.sent else FG2)
    if beat.sent:
        text(d, 40, 208, f"{len(upload.words)} words", small, FG2)
        text(d, 40, 236, "offline square", small, FG2)
        text(d, 40, 313, beat.packet or "request received", small, FG2)
        if beat.rx is not None:
            text(d, 40, 349, f"0x{beat.rx:04x}", title, BLUE)
    text(d, 377, 173, "TX", label, FG2)
    text(d, 377, 239, "RX", label, FG2)
    arrow(d, 351, 285, 173, ACCENT if beat.tx is not None else LINE)
    arrow(d, 285, 351, 239, BLUE if beat.rx is not None else LINE)
    raster(d, 479, 173, beat.tx, beat.slots, ACCENT)
    raster(d, 479, 239, beat.rx, beat.slots, BLUE)
    text(d, 479, 268, "start", small, FG3, "mm")
    text(d, 829, 268, "bit 13", small, FG3, "mm")
    text(d, 377, 323, "Receive latch", title)
    valid = beat.rx is not None and beat.slots == BITS + 1
    received = (beat.rx or 0) & ((1 << max(0, beat.slots - 1)) - 1)
    squares(d, [(received >> i) & 1 for i in range(BITS)], 479, 354, 25, 17, on=BLUE)
    text(d, 479, 395, "bit 0", small, FG2, "mm")
    text(d, 900, 362, "valid", small, BLUE if valid else FG3, "rm")

    text(d, 966, 108, "Program RAM", title)
    text(d, 1539, 109, f"{beat.writes:2}/{len(upload.words)} words", small, FG2, "rm")
    x, y, row_h = 1072, 146, 177 / len(upload.words)
    text(d, x - 14, y + 3, upload.trace["entry_address"], small, FG2, "rm")
    text(d, x - 14, y + (len(upload.words) - 1) * row_h, upload.trace["writes"][-1]["address"], small, FG2, "rm")
    for j, word in enumerate(upload.words):
        yy = y + j * row_h
        squares(d, bits(word if j < beat.writes else 0, 32), x, yy, 13.5, row_h - 1.6)
    if beat.writes:
        yy = y + (beat.writes - 1) * row_h
        line(d, [(x - 8, yy), (x - 8, yy + row_h - 1)], ACCENT, 3)
    line(d, [(964, 335), (1540, 335)])
    text(d, 966, 368, "PC", label, FG2)
    text(d, 1012, 368, beat.pc, title, ACCENT if beat.pc != "bootloader" else FG2)
    if beat.executed:
        text(d, 1128, 368, f"t0 = {upload.trace['t0_after_entry_instruction']}", label)
    elif beat.pc == upload.trace["entry_address"]:
        text(d, 1128, 368, "lui t0, 0x80000", label)
    return img.resize((1280, 368), Image.Resampling.LANCZOS)


def bootloader(trace):
    upload = Upload(trace)
    assert upload.words[0] == 0x800002B7  # lui t0, 0x80000
    fonts = (font(MONO, 17), font(MONO, 20), font(DISPLAY, 26))
    frames = [render(upload, upload.at(i * .08), fonts) for i in range(94)]
    sheet = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for i, frame in enumerate((frames[8], frames[36], frames[65], frames[-1])):
        sheet.paste(frame, (0, i * frame.height))
    palette = sheet.quantize(240, method=Image.Quantize.MEDIANCUT)
    # Preserve the wire colors even though spike marks occupy very few pixels.
    colors = palette.getpalette()[:240 * 3]
    colors.extend(channel for color in (ACCENT, BLUE, FG, FG2, FG3, LINE, PANEL, GROUND) for channel in color)
    palette.putpalette(colors)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(OUT / "bootloader.gif", save_all=True, append_images=frames[1:],
                   duration=[80] * (len(frames) - 1) + [430], loop=0, optimize=True)


if __name__ == "__main__":
    trace = json.loads((OUT / "boot_trace.json").read_text())
    compilation(trace)
    bootloader(trace)
