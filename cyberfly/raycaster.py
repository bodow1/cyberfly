MAZE = [
    0xFFFF, 0b1000000000000001, 0b1011100001110001, 0b1010000000010001, 0b1010001100010001,
    0b1000001100000001, 0b1000000000011101, 0b1001110000000101, 0b1000000000000101, 0b1000000011000001,
    0b1011000011000001, 0b1001000000001101, 0b1001000000001001, 0b1000000000000001, 0xFFFF, 0xFFFF,
]
COS = [64, 64, 63, 61, 59, 57, 54, 50, 45, 40, 36, 30, 24, 19, 12, 6]
COLUMNS = 32
MAX_STEPS = 31


def wall(x: int, y: int) -> int:
    return (MAZE[y & 15] >> (x & 15)) & 1


def cosq(a: int) -> int:
    a &= 63
    q, k = a >> 4, a & 15
    return [COS[k], -COS[16 - k] if k else 0, -COS[k], COS[16 - k] if k else 0][q]


def sinq(a: int) -> int:
    return cosq((a - 16) & 63)


def render(px: int, py: int, ang: int) -> list[int]:
    out = []
    for c in range(COLUMNS):
        ra = (ang + 16 - c) & 63
        dx, dy = cosq(ra), sinq(ra)
        x, y, q = px, py, 0
        while True:
            x, y, q = (x + dx) & 0xFFF, (y + dy) & 0xFFF, q + 1
            if wall(x >> 8, y >> 8) or q == MAX_STEPS:
                break
        out.append(q)
    return out


class Raycaster:
    def __init__(self):
        self.cam = 0
        self.dist = [MAX_STEPS] * COLUMNS
        self.demon = 0

    def camera(self, value: int):
        self.cam = value

    def draw(self, value: int):
        if not value >> 31:
            return
        self.dist = render(self.cam & 0xFFF, (self.cam >> 12) & 0xFFF, (self.cam >> 24) & 63)
        lo, hi, depth = value & 31, (value >> 5) & 31, (value >> 10) & 31
        self.demon = sum(1 << c for c in range(COLUMNS) if lo <= c <= hi) if depth else 0

    def aux(self) -> int:
        return self.dist[15]
