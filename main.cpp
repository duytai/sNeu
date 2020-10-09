#include "fuzzer.h"
#include "debug.h"

#include <stdio.h>
#include <signal.h>
#include <string.h>

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

void show_usage(char* name) {
  SAYF("Usage: %s -i <in_dir> <app>\n", name);
  exit(EXIT_SUCCESS);
}

int main(int argc, char* argv[]) {
  setup_signal_handlers();

  fuzzer.parse_arguments(argc, argv);
  fuzzer.setup_fds();
  fuzzer.setup_shm();
  fuzzer.init_forkserver();
}
