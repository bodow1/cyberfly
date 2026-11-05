unsigned __mulsi3(unsigned a, unsigned b) {
  unsigned r = 0;
  for (; b; b >>= 1, a <<= 1)
    if (b & 1) r += a;
  return r;
}

unsigned __udivsi3(unsigned n, unsigned d) {
  unsigned q = 0, r = 0;
  for (int i = 31; i >= 0; i--) {
    r = (r << 1) | ((n >> i) & 1);
    if (r >= d) { r -= d; q |= 1u << i; }
  }
  return q;
}

unsigned __umodsi3(unsigned n, unsigned d) { return n - __mulsi3(__udivsi3(n, d), d); }

int __divsi3(int a, int b) {
  unsigned q = __udivsi3(a < 0 ? -a : a, b < 0 ? -b : b);
  return (a < 0) ^ (b < 0) ? -(int)q : (int)q;
}

int __modsi3(int a, int b) { return a - (int)__mulsi3((unsigned)__divsi3(a, b), (unsigned)b); }

void *memset(void *p, int c, unsigned n) {
  unsigned char *q = p;
  while (n--) *q++ = c;
  return p;
}

void *memcpy(void *d, const void *s, unsigned n) {
  unsigned char *a = d;
  const unsigned char *b = s;
  while (n--) *a++ = *b++;
  return d;
}
