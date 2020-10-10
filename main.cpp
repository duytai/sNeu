#include "fuzzer.h"
#include "debug.h"

#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

Fuzzer fuzzer;

FuzzerOpt parse_arguments(int argc, char* argv[]) {
  FuzzerOpt fuzzer_opt;
  int i = 1;

  if (strcmp(argv[argc - 1], "@@") == 0) {
    fuzzer_opt.use_stdin = false;
    argv[argc - 1] = fuzzer_opt.out_file;
  }

  while (i < argc - 1) {
    char* opt = argv[i];
    if (strcmp(opt, "-i") == 0 && argv[i + 1] != NULL) {
      fuzzer_opt.in_dir = argv[i + 1];
      i += 2;
    } else break;
  }

  fuzzer_opt.target_argv = argv + i;

  if (fuzzer_opt.in_dir == NULL || fuzzer_opt.target_argv[0] == NULL) {
    SAYF("  Usage: %s <in_dir> <app>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  return fuzzer_opt;
}


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

int main(int argc, char* argv[]) {
  char* input = "hello world";
  u32 exec_tmout = EXEC_TIMEOUT;
  setup_signal_handlers();
  fuzzer.load_opt(parse_arguments(argc, argv));

  fuzzer.write_to_testcase(input, strlen(input));
  u8 v = fuzzer.run_target(exec_tmout);
  OKF("V is %d", v);
  OKF("new_bis: %d", fuzzer.has_new_bits());
  OKF("new_bis: %d", fuzzer.has_new_bits());
}
