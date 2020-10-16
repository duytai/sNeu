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
#include <set>

#include <libfuzzer/fuzzer.h>
#include <libfuzzer/util.h>

/* X86 only */

enum {
  /* 00 */ FAULT_NONE,
  /* 01 */ FAULT_TMOUT,
  /* 02 */ FAULT_CRASH,
  /* 03 */ FAULT_ERROR,
  /* 04 */ FAULT_NOINST,
  /* 05 */ FAULT_NOBITS
};

Fuzzer::Fuzzer(void) {
  this->init_count_class16();
  this->setup_fds();
  this->setup_shm();
  this->start_time = get_cur_time();
}

void Fuzzer::load_opt(SNeuOptions opt) {
  this->opt = opt;
  this->init_forkserver();
}

void Fuzzer::classify_counts() {
  u64* mem = (u64*) this->trace_bits;
  u32 i = MAP_SIZE >> 3;
  while (i--) {
    /* Optimize for sparse bitmaps. */
    if (unlikely(*mem)) {
      u16* mem16 = (u16*)mem;
      mem16[0] = count_class_lookup16[mem16[0]];
      mem16[1] = count_class_lookup16[mem16[1]];
      mem16[2] = count_class_lookup16[mem16[2]];
      mem16[3] = count_class_lookup16[mem16[3]];
    }
    mem++;
  }
}

void Fuzzer::init_count_class16(void) {
  u32 b1, b2;
  for (b1 = 0; b1 < 256; b1++)
    for (b2 = 0; b2 < 256; b2++)
      this->count_class_lookup16[(b1 << 8) + b2] =
        (this->count_class_lookup8[b1] << 8) |
        this->count_class_lookup8[b2];
}

void Fuzzer::setup_fds(void) {
  unlink(this->opt.out_file);
  this->out_fd = open(this->opt.out_file, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (this->out_fd < 0) PFATAL("Unable to create '%s'", this->opt.out_file);

  this->dev_null_fd = open("/dev/null", O_RDWR);
  if (this->dev_null_fd < 0) PFATAL("Unable to open /dev/null");
}

void Fuzzer::remove_shm(void) {
  shmctl(this->shm_id, IPC_RMID, NULL);
}

void Fuzzer::setup_shm(void) {
  /*
   * [0, MAP_SIZE] => trace_bits
   * [MAP_SIZE, 2 * MAP_SIZE] => branch distance
   * */
  this->shm_id = shmget(IPC_PRIVATE, MAP_SIZE * 2, IPC_CREAT | IPC_EXCL | 0600);
  if (shm_id < 0) PFATAL("shmget() failed");

  setenv(SHM_ENV_VAR, std::to_string(shm_id).c_str(), 1);
  memset(this->virgin_bits, 255, MAP_SIZE);
  memset(this->virgin_loss, 255, MAP_SIZE);

  void* trace_bits = shmat(shm_id, NULL, 0);
  if (trace_bits == (void *)-1) PFATAL("shmat() failed");
  this->trace_bits = (u8*) trace_bits;
  this->loss_bits = (u8*) trace_bits + MAP_SIZE;
  this->loss_bits[0] = 255;
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

    if (this->opt.use_stdin) {
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

    execv(this->opt.target_argv[0], this->opt.target_argv);
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

u8 Fuzzer::run_target(vector<char>& mem, u32 timeout) {

  static struct itimerval it;
  static int prev_timed_out = 0;
  int status, res, kill_signal;

  lseek(this->out_fd, 0, SEEK_SET);
  write(this->out_fd, mem.data(), mem.size());
  ftruncate(this->out_fd, mem.size());
  lseek(this->out_fd, 0, SEEK_SET);

  memset(this->trace_bits, 0, MAP_SIZE);
  memset(this->loss_bits, 255, MAP_SIZE);
  MEM_BARRIER();
  this->child_timed_out = 0;

  if ((res = write(this->fsrv_ctl_fd, &prev_timed_out, 4)) != 4) {
    FATAL("Unable to request new process from fork server (OOM?)");
  }
  if ((res = read(this->fsrv_st_fd, &this->child_pid, 4)) != 4) {
    FATAL("Unable to request new process from fork server (OOM?)");
  }
  if (this->child_pid <= 0) FATAL("Fork server is misbehaving (OOM?)");

  it.it_value.tv_sec = (timeout / 1000);
  it.it_value.tv_usec = (timeout % 1000) * 1000;
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

  this->classify_counts();
  this->update_loss();

  vector<u8> loss_bits(this->loss_bits, this->loss_bits + MAP_SIZE);
  this->tc = { .buffer = mem, .loss_bits = loss_bits, .hnb = this->has_new_bits() };
  this->total_ints += this->tc.hnb > 0 ? 1 : 0;
  this->show_info(0);

  if (WIFSIGNALED(status)) {
    kill_signal = WTERMSIG(status);
    if (this->child_timed_out && kill_signal == SIGKILL) return FAULT_TMOUT;
    return FAULT_CRASH;
  }

  return FAULT_NONE;
}

u8 Fuzzer::has_new_bits(void) {
  u64* current = (u64*) this->trace_bits;
  u64* virgin  = (u64*) this->virgin_bits;
  u32 i = (MAP_SIZE >> 3);
  u8 ret = 0;

  while (i--) {
    if (unlikely(*current) && unlikely(*current & *virgin)) {
      if (likely(ret < 2)) {
        u8* cur = (u8*)current;
        u8* vir = (u8*)virgin;

        /* Looks like we have not found any new bytes yet; see if any non-zero
           bytes in current[] are pristine in virgin[]. */
        if ((cur[0] && vir[0] == 0xff) || (cur[1] && vir[1] == 0xff) ||
            (cur[2] && vir[2] == 0xff) || (cur[3] && vir[3] == 0xff) ||
            (cur[4] && vir[4] == 0xff) || (cur[5] && vir[5] == 0xff) ||
            (cur[6] && vir[6] == 0xff) || (cur[7] && vir[7] == 0xff)) ret = 2;
        else ret = 1;
      }
      *virgin &= ~*current;
    }
    current++;
    virgin++;
  }

  return ret;
}

void Fuzzer::update_loss(void) {
  u64* current = (u64*) this->loss_bits;
  u64* virgin  = (u64*) this->virgin_loss;
  u32 i = (MAP_SIZE >> 3);

  while (i--) {
    if (unlikely(*current) || likely(*current & *virgin)) {
      *virgin &= *current;
    }
    current ++;
    virgin ++;
  }
}

void Fuzzer::handle_stop_sig(void) {
  if (this->child_pid > 0) kill(this->child_pid, SIGKILL);
  if (this->forksrv_pid > 0) kill(this->forksrv_pid, SIGKILL);
}

#define UP "\x1b[A"
#define DOWN "\n"
void Fuzzer::show_info(u8 force) {
  u64 duration = (get_cur_time() - this->start_time) / 1000;
  if (this->total_execs == 1) SAYF(DOWN DOWN DOWN DOWN);
  if (duration != this->total_time || force) {
    SAYF(UP UP UP UP);
    SAYF("  Execs\t: %llu/%llu\n", this->total_ints, this->total_execs);
    SAYF("  Speed\t: %llu\n", this->total_execs / (duration + 1));
    SAYF("  Queue\t: %d/%d\n", this->queue_idx, this->queue_size);
    SAYF("  Stage\t: %s\n", this->stage);
    this->total_time = duration;
  }
}

Fuzzer::~Fuzzer() {
  this->remove_shm();
}
