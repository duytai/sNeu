#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "fuzzer.h"

void Fuzzer::setup_fds(void) {
  char* fname = ".cur_input";
  unlink(fname);
  this->out_fd = open(fname, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (this->out_fd < 0) FATAL("Unable to create '%s'", fname);
}
