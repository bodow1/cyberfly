#define IO (*(volatile unsigned *)0x80000000u)
#define CALL_CPU (1u << 3)
#define CALL_LLM (1u << 10)
#define SIGN_LEFT (1u << 4)
#define SIGN_RIGHT (1u << 5)
#define SIGN_LETTER (1u << 9)
#define REWARD_CPU (1u << 12)
#define REWARD_LLM (1u << 13)
#define ACK_LLM (1u << 29)
#define ACK_CPU (1u << 30)
#define LEFT 0x00Fu
#define RIGHT 0x0F0u

int main(void) {
  for (;;) {
    unsigned v = IO;
    if (v & CALL_CPU) {
      if (v & SIGN_LEFT) IO = ACK_CPU | REWARD_CPU | LEFT;
      else if (v & SIGN_RIGHT) IO = ACK_CPU | REWARD_CPU | RIGHT;
      else IO = ACK_CPU;
      IO = 0;
    } else if (v & CALL_LLM) {
      if (v & SIGN_LETTER) IO = ACK_LLM | REWARD_LLM;
      else IO = ACK_LLM;
      IO = 0;
    }
  }
}
