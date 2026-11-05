#define IO  (*(volatile unsigned *)0x80000000u)
#define AUX (*(volatile unsigned *)0x80000004u)
#define CAM (*(volatile unsigned *)0x80000004u)
#define DEM (*(volatile unsigned *)0x80000008u)

static const unsigned char HALF_WIDTH[32] = {9, 9, 9, 9, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 3,
                                             3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1};
static const signed char COS[16] = {64, 64, 63, 61, 59, 57, 54, 50, 45, 40, 36, 30, 24, 19, 12, 6};

static int cosq(int a) {
  a &= 63;
  int q = a >> 4, k = a & 15;
  if (q == 0) return COS[k];
  if (q == 1) return k ? -COS[16 - k] : 0;
  if (q == 2) return -COS[k];
  return k ? COS[16 - k] : 0;
}

int main(void) {
  int px = 0x780, py = 0x880, ang = 0;
  int bearing = 9, depth = 28;
  int score = 0, hp = 3, flags = 0;
  for (;;) {
    int c = 16 - bearing, hw = HALF_WIDTH[depth & 31];
    int lo = c - hw, hi = c + hw;
    if (bearing > 16 || bearing < -15) { lo = 31; hi = 0; }
    if (lo < 0) lo = 0;
    if (hi > 31) hi = 31;
    CAM = (unsigned)px | (unsigned)py << 12 | (unsigned)(ang & 63) << 24;
    DEM = (unsigned)lo | (unsigned)hi << 5 | (unsigned)(depth & 31) << 10 | 1u << 31;
    while (AUX >> 31) {
    }
    int ahead = AUX & 31;
    IO = (unsigned)score | (unsigned)hp << 4 | (unsigned)(bearing & 63) << 8 | (unsigned)depth << 14 |
         (unsigned)(ang & 63) << 19 | (unsigned)flags << 25 | (unsigned)(px >> 8) << 28;
    int left = 0, right = 0, fire = 0;
    for (int i = 0; i < 4; i++) {
      unsigned v = IO;
      left += v & 1;
      right += (v >> 1) & 1;
      fire += (v >> 2) & 1;
    }
    flags = 0;
    if (left > right) { ang++; bearing--; }
    else if (right > left) { ang--; bearing++; }
    if (fire) {
      flags |= 1;
      if (bearing >= -1 && bearing <= 1) {
        flags |= 2;
        score++;
        bearing = (score & 1) ? 10 : -10;
        depth = 28;
      }
    }
    if (ahead > 6) {
      px += cosq(ang) >> 2;
      py += cosq(ang - 16) >> 2;
    }
    if (--depth <= 2) {
      flags |= 4;
      hp--;
      depth = 28;
      bearing = -bearing;
      if (hp == 0) { IO = 1u << 31; return score; }
    }
    if (score == 9) { IO = 1u << 31 | 9; return score; }
  }
}
