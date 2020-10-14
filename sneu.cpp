#include <libfuzzer/fuzzer.h>
#include <libfuzzer/testsuit.h>
#include <libfuzzer/debug.h>
#include <libutil/util.h>

#include <signal.h>
#include <string.h>

using namespace std;

Fuzzer fuzzer;

SNeuOptions parse_arguments(int argc, char* argv[]) {
  SNeuOptions sneu_opt;
  int i = 1;

  if (strcmp(argv[argc - 1], "@@") == 0) {
    sneu_opt.use_stdin = false;
    argv[argc - 1] = sneu_opt.out_file;
  }

  while (i < argc - 1) {
    char* opt = argv[i];
    if (strcmp(opt, "-i") == 0 && argv[i + 1] != NULL) {
      sneu_opt.in_dir = argv[i + 1];
      i += 2;
    } else break;
  }

  sneu_opt.target_argv = argv + i;

  if (sneu_opt.in_dir == NULL || sneu_opt.target_argv[0] == NULL) {
    SAYF("  Usage: %s -i <in_dir> <app>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  return sneu_opt;
}

void handle_timeout(int) {
  fuzzer.handle_timeout();
}

void handle_stop_sig(int) {
  fuzzer.handle_stop_sig();
}

void setup_signal_handlers() {
  struct sigaction sa;

  sa.sa_handler   = NULL;
  sa.sa_flags     = SA_RESTART;
  sa.sa_sigaction = NULL;

  sigemptyset(&sa.sa_mask);

  /* Various ways of saying "stop". */

  sa.sa_handler = handle_stop_sig;
  sigaction(SIGHUP, &sa, NULL);
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);

  /* Exec timeout notifications. */

  sa.sa_handler = handle_timeout;
  sigaction(SIGALRM, &sa, NULL);
}

int main(int argc, char* argv[]) {
  auto opt = parse_arguments(argc, argv);
  setup_signal_handlers();
  TestSuite suite(&fuzzer, opt);
  suite.load_from_in_dir();
  suite.exec_remaining();
  suite.smart_mutate();
}
