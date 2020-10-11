#include "fuzzer.h"
#include "mutator.h"
#include "debug.h"

#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;
using namespace std::filesystem;

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
  auto mut = Mutator(&fuzzer);
  auto opt = parse_arguments(argc, argv);
  u32 exec_tmout = EXEC_TIMEOUT;
  setup_signal_handlers();
  fuzzer.load_opt(opt);

  vector<directory_entry> files(directory_iterator(opt.in_dir), directory_iterator());
  sort(files.begin(), files.end());

  for (auto &file: files) {
    if (file.is_regular_file() && file.file_size() > 0) {
      ifstream st(file.path(), ios::binary);
      vector<char> buffer((istreambuf_iterator<char>(st)), istreambuf_iterator<char>());
      fuzzer.run_target(buffer, exec_tmout);
      printf("HNB: %d\n", fuzzer.hnb);
    }
  }
  cout << "total_execs " << fuzzer.total_execs << endl;
  // mut.mutate();
}
