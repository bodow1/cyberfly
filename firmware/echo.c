#include "radio.h"

int main(void) {
  for (;;) radio_send(radio_recv(0) + 1);
}
