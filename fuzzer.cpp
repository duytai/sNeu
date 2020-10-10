#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string>

#include "fuzzer.h"

enum {
  /* 00 */ FAULT_NONE,
  /* 01 */ FAULT_TMOUT,
  /* 02 */ FAULT_CRASH,
  /* 03 */ FAULT_ERROR,
  /* 04 */ FAULT_NOINST,
  /* 05 */ FAULT_NOBITS
};

Fuzzer::Fuzzer(void) {
}

void Fuzzer::setup_fds(void) {
  unlink(this->out_file);
  this->out_fd = open(this->out_file, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (this->out_fd < 0) PFATAL("Unable to create '%s'", this->out_file);

  this->dev_null_fd = open("/dev/null", O_RDWR);
  if (this->dev_null_fd < 0) PFATAL("Unable to open /dev/null");
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

void Fuzzer::parse_arguments(int argc, char* argv[]) {
  int i = 1;

  if (strcmp(argv[argc - 1], "@@") == 0) {
    this->use_stdin = false;
    argv[argc - 1] = this->out_file;
  }

  while (i < argc - 1) {
    char* opt = argv[i];
    if (strcmp(opt, "-i") == 0 && argv[i + 1] != NULL) {
      this->in_dir = argv[i + 1];
      i += 2;
    } else break;
  }

  this->target_argv = argv + i;

  if (this->in_dir == NULL || this->target_argv[0] == NULL) {
    SAYF("  Usage: %s <in_dir> <app>\n", argv[0]);
    exit(EXIT_SUCCESS);
  }
}

void Fuzzer::init_forkserver(void) {

  static struct itimerval it;
  int st_pipe[2], ctl_pipe[2];
  int status, rlen;

  ACTF("Spinning up the fork server...");
  if (pipe(st_pipe) || pipe(ctl_pipe)) PFATAL("pipe() failed");

  this->forksrv_pid = fork();
  if (this->forksrv_pid < 0) PFATAL("fork() failed");

  if (!this->forksrv_pid) {
    setsid();

    dup2(this->dev_null_fd, 1);
    dup2(this->dev_null_fd, 2);

    if (this->use_stdin) {
      dup2(this->out_fd, 0);
      close(this->out_fd);
    } else {
      dup2(this->dev_null_fd, 0);
    }

    if (dup2(ctl_pipe[0], FORKSRV_FD) < 0) PFATAL("dup2() failed");
    if (dup2(st_pipe[1], FORKSRV_FD + 1) < 0) PFATAL("dup2() failed");

    close(ctl_pipe[0]);
    close(ctl_pipe[1]);
    close(st_pipe[0]);
    close(st_pipe[1]);

    execv(this->target_argv[0], this->target_argv);
    exit(EXIT_SUCCESS);
  }

  close(ctl_pipe[0]);
  close(st_pipe[1]);

  this->fsrv_ctl_fd = ctl_pipe[1];
  this->fsrv_st_fd = st_pipe[0];

  it.it_value.tv_sec = ((EXEC_TIMEOUT * FORK_WAIT_MULT) / 1000);
  it.it_value.tv_usec = ((EXEC_TIMEOUT * FORK_WAIT_MULT) % 1000) * 1000;

  setitimer(ITIMER_REAL, &it, NULL);

  rlen = read(fsrv_st_fd, &status, 4);

  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;

  setitimer(ITIMER_REAL, &it, NULL);

  if (rlen == 4) {
    OKF("All right - fork server is up.");
    return;
  }

  if (waitpid(this->forksrv_pid, &status, 0) <= 0) PFATAL("waitpid() failed");
  PFATAL("Fork server handshake failed");
}

void Fuzzer::write_to_testcase(char* mem, u32 len) {
  lseek(this->out_fd, 0, SEEK_SET);
  if (write(this->out_fd, mem, len) != len) PFATAL("write() failed");
  if (ftruncate(this->out_fd, len)) PFATAL("ftruncate() failed");
  lseek(this->out_fd, 0, SEEK_SET);
}

u8 Fuzzer::run_target(u32 exec_tmout) {

  static struct itimerval it;
  static int prev_timed_out = 0;
  int status, res, kill_signal;

  memset(this->trace_bits, 0, MAP_SIZE);
  MEM_BARRIER();
  this->child_timed_out = 0;

  if ((res = write(this->fsrv_ctl_fd, &prev_timed_out, 4)) != 4) {
    FATAL("Unable to request new process from fork server (OOM?)");
  }
  if ((res = read(this->fsrv_st_fd, &this->child_pid, 4)) != 4) {
    FATAL("Unable to request new process from fork server (OOM?)");
  }
  if (this->child_pid <= 0) FATAL("Fork server is misbehaving (OOM?)");

  it.it_value.tv_sec = (exec_tmout / 1000);
  it.it_value.tv_usec = (exec_tmout % 1000) * 1000;
  setitimer(ITIMER_REAL, &it, NULL);

  if ((res = read(this->fsrv_st_fd, &status, 4)) != 4) { // get return status
    FATAL("Unable to communicate with fork server (OOM?)");
  }

  if (!WIFSTOPPED(status)) this->child_pid = 0;

  getitimer(ITIMER_REAL, &it);
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;
  setitimer(ITIMER_REAL, &it, NULL);

  prev_timed_out = this->child_timed_out;
  this->total_execs ++;
  MEM_BARRIER();

  if (WIFSIGNALED(status)) {
    kill_signal = WTERMSIG(status);
    if (this->child_timed_out && kill_signal == SIGKILL) return FAULT_TMOUT;
    return FAULT_CRASH;
  }

  return FAULT_NONE;
}

Fuzzer::~Fuzzer() {
  this->remove_shm();
}
