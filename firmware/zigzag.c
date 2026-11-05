#define IO (*(volatile unsigned *)0x80000000u)
#define LEFT 0x00Fu
#define RIGHT 0x0F0u

static void hold(unsigned bits, int ticks) {
  IO = bits;
  for (volatile int i = 0; i < ticks; i++) {
  }
}

int main(void) {
  for (;;) {
    hold(0, 2);
    hold(LEFT, 1);
    hold(0, 2);
    hold(RIGHT, 1);
  }
}
