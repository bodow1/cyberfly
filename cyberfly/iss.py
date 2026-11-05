from collections.abc import Callable

MASK = 0xFFFFFFFF


def sext(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


class ISS:
    def __init__(self, program: list[int], aw: int = 8, io_in: Callable[[], int] | int = 0, gpu=None,
                 on_store: Callable[[int], None] | None = None):
        self.mem = list(program) + [0] * ((1 << aw) - len(program))
        self.aw = aw
        self.r = [0] * 32
        self.pc = 0
        self.halted = False
        self.io_out: list[int] = []
        self.io_in = io_in
        self.gpu = gpu
        self.on_store = on_store
        self.steps = 0

    def _word(self, a: int) -> int:
        return self.mem[(a >> 2) & ((1 << self.aw) - 1)]

    def _load_io(self, a: int) -> int:
        if self.gpu is not None and a & 4:
            return self.gpu.aux()
        return self.io_in() if callable(self.io_in) else self.io_in

    def _store_io(self, a: int, v: int):
        if self.gpu is not None and (a >> 2) & 3 == 1:
            self.gpu.camera(v)
        elif self.gpu is not None and (a >> 2) & 3 == 2:
            self.gpu.draw(v)
        else:
            self.io_out.append(v)
            if self.on_store is not None:
                self.on_store(v)

    def step(self):
        if self.halted:
            return
        ir = self._word(self.pc)
        op, rd, f3 = ir & 0x7F, (ir >> 7) & 31, (ir >> 12) & 7
        r1, r2 = self.r[(ir >> 15) & 31], self.r[(ir >> 20) & 31]
        f7 = (ir >> 30) & 1
        imm_i = sext(ir >> 20, 12)
        imm_s = sext(((ir >> 25) << 5) | ((ir >> 7) & 31), 12)
        imm_b = sext((((ir >> 31) & 1) << 12) | (((ir >> 7) & 1) << 11) | (((ir >> 25) & 63) << 5)
                     | (((ir >> 8) & 15) << 1), 13)
        imm_j = sext((((ir >> 31) & 1) << 20) | (((ir >> 12) & 255) << 12) | (((ir >> 20) & 1) << 11)
                     | (((ir >> 21) & 1023) << 1), 21)
        npc = (self.pc + 4) & MASK
        wb = None
        if op == 0x37:
            wb = ir & 0xFFFFF000
        elif op == 0x17:
            wb = (self.pc + (ir & 0xFFFFF000)) & MASK
        elif op == 0x6F:
            wb, npc = npc, (self.pc + imm_j) & MASK
        elif op == 0x67:
            wb, npc = npc, (r1 + imm_i) & MASK & ~1
        elif op == 0x63:
            s1, s2 = sext(r1, 32), sext(r2, 32)
            if [r1 == r2, r1 != r2, False, False, s1 < s2, s1 >= s2, r1 < r2, r1 >= r2][f3]:
                npc = (self.pc + imm_b) & MASK
        elif op == 0x03:
            a = (r1 + imm_i) & MASK
            w = (self._load_io(a) if a >> 31 else self._word(a)) >> (8 * (a & 3))
            wb = {0: sext(w, 8) & MASK, 1: sext(w, 16) & MASK, 4: w & 0xFF, 5: w & 0xFFFF}.get(f3, w & MASK)
        elif op == 0x23:
            a = (r1 + imm_s) & MASK
            if a >> 31:
                self._store_io(a, r2)
            else:
                i, sh = (a >> 2) & ((1 << self.aw) - 1), 8 * (a & 3)
                m = {0: 0xFF << sh, 1: 0xFFFF << sh}.get(f3, MASK) & MASK
                self.mem[i] = (self.mem[i] & ~m & MASK) | ((r2 << sh) & m)
        elif op in (0x13, 0x33):
            b = r2 if op == 0x33 else imm_i & MASK
            sh = b & 31
            wb = [
                (r1 - b if op == 0x33 and f7 else r1 + b),
                r1 << sh,
                int(sext(r1, 32) < sext(b, 32)),
                int(r1 < b),
                r1 ^ b,
                (sext(r1, 32) >> sh) if f7 else r1 >> sh,
                r1 | b,
                r1 & b,
            ][f3] & MASK
        elif op == 0x73:
            self.halted = True
        if wb is not None and rd:
            self.r[rd] = wb & MASK
        self.pc = npc
        self.steps += 1

    def run(self, max_steps: int = 1_000_000) -> "ISS":
        while not self.halted and self.steps < max_steps:
            self.step()
        return self
