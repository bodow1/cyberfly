#define OUT (*(volatile unsigned *)0x80000000u)

static int prime(unsigned n) {
  if (n < 2) return 0;
  for (unsigned d = 2; d * d <= n; d++) {
    unsigned r = n;
    while (r >= d) r -= d;
    if (r == 0) return 0;
  }
  return 1;
}

int main(void) {
  unsigned a = 0, b = 1;
  for (int i = 0; i < 12; i++) {
    OUT = a;
    unsigned t = a + b;
    a = b;
    b = t;
  }
  for (unsigned n = 2; n < 40; n++)
    if (prime(n)) OUT = 1000 + n;
  return 0;
}
