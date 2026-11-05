#define IO (*(volatile unsigned *)0x80000000u)
#define CALL_CPU (1u << 3)
#define SIGN_LEFT (1u << 4)
#define SIGN_RIGHT (1u << 5)
#define SIGN_LETTER (1u << 9)
#define CALL_LLM (1u << 10)
#define LEFT 0x00Fu
#define RIGHT 0x0F0u
#define REWARD_CPU (1u << 12)
#define REWARD_LLM (1u << 13)
#define ACK_LLM (1u << 29)
#define ACK_CPU (1u << 30)

static void answer(unsigned bits) {
  IO = bits;
  IO = 0;
}

int main(void) {
  for (;;) {
    unsigned v = IO;
    if (!(v & (CALL_CPU | CALL_LLM))) continue;
    if (v & CALL_CPU) {
      if (v & SIGN_LEFT) answer(ACK_CPU | REWARD_CPU | LEFT);
      else if (v & SIGN_RIGHT) answer(ACK_CPU | REWARD_CPU | RIGHT);
      else answer(ACK_CPU);
    }
    if (v & CALL_LLM) answer(v & SIGN_LETTER ? ACK_LLM | REWARD_LLM : ACK_LLM);
  }
}
