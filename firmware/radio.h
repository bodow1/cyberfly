#define IO (*(volatile unsigned *)0x80000000u)
#define FRAME 0x3FFFu
#define SEND (1u << 31)
#define ACK (1u << 30)

static inline void radio_send(unsigned w) { IO = SEND | (w & FRAME); }

static inline int radio_ready(void) { return (int)IO < 0; }

__attribute__((noinline)) static unsigned radio_recv(unsigned keep) {
  unsigned v;
  while ((int)(v = IO) >= 0) {
  }
  IO = ACK | keep;
  while ((int)IO < 0) {
  }
  return v & FRAME;
}
