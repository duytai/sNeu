#include "fuzzer.h"
#include "debug.h"

#include <stdio.h>
#include <signal.h>

Fuzzer fuzzer;

void handle_timeout(int) {
  fuzzer.handle_timeout();
}

void setup_signal_handlers() {
  struct sigaction sa;

  sa.sa_handler   = NULL;
  sa.sa_flags     = SA_RESTART;
  sa.sa_sigaction = NULL;

  sigemptyset(&sa.sa_mask);

  sa.sa_handler = handle_timeout;
  sigaction(SIGALRM, &sa, NULL);
}

int main() {
  setup_signal_handlers();
  fuzzer.setup_fds();
  fuzzer.setup_shm();
}
