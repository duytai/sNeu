#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <signal.h>
#include <string>

#include "fuzzer.h"

void Fuzzer::setup_fds(void) {
  char* fname = ".cur_input";

  unlink(fname);
  this->out_fd = open(fname, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (this->out_fd < 0) PFATAL("Unable to create '%s'", fname);
}

void Fuzzer::remove_shm(void) {
  shmctl(this->shm_id, IPC_RMID, NULL);
}

void Fuzzer::setup_shm(void) {
  this->shm_id = shmget(IPC_PRIVATE, MAP_SIZE, IPC_CREAT | IPC_EXCL | 0600);
  if (shm_id < 0) PFATAL("shmget() failed");

  setenv(SHM_ENV_VAR, std::to_string(shm_id).c_str(), 1);
  memset(virgin_bits, 255, MAP_SIZE);

  void* trace_bits = shmat(shm_id, NULL, 0);
  if (trace_bits == (void *)-1) PFATAL("shmat() failed");
  this->trace_bits = (u8*) trace_bits;
}

void Fuzzer::handle_timeout(void) {
  if (this->child_pid > 0) {
    this->child_timed_out = 1;
    kill(this->child_pid, SIGKILL);
  } else if (this->child_pid == -1 && this->forksrv_pid > 0) {
    this->child_timed_out = 1;
    kill(this->forksrv_pid, SIGKILL);
  }
}

Fuzzer::~Fuzzer() {
  this->remove_shm();
}
