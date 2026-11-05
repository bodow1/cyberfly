#include "radio.h"

#define LEFT 0x00Fu
#define RIGHT 0x0F0u
#define STAND 0xF00u
#define LOADING (1u << 24)
#define BAD (0xBADu << 16)
#define ORIGIN 0x200u
#define MAX_WORDS 96u

__attribute__((noinline)) static void hold(unsigned bits, int ticks) {
  IO = bits;
  for (volatile int i = 0; i < ticks; i++) {
  }
}

__attribute__((noinline)) static unsigned next(void) { return radio_recv(STAND | LOADING); }

int main(void) {
  hold(LEFT, 1);
  hold(0, 3);
  hold(RIGHT, 2);
  hold(0, 3);
  hold(LEFT, 2);
  hold(0, 3);
  IO = STAND;
  radio_send('?');
  for (;;) {
    IO = STAND;
    unsigned head = radio_recv(STAND);
    unsigned n = head & 0xFFFu;
    if (head >> 12 != 3 || n > MAX_WORDS) continue;
    volatile unsigned *dst = (volatile unsigned *)ORIGIN;
    unsigned sum = 0;
    for (unsigned i = 0; i < n; i++) {
      unsigned a = next(), b = next(), c = next();
      sum += a + b + c;
      dst[i] = a | b << 11 | c << 22;
    }
    if (next() == (sum & FRAME)) ((void (*)(void))ORIGIN)();
    IO = STAND | BAD;
  }
}
