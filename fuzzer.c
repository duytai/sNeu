#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <dirent.h>
#include <string.h>
#include <stdint.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define MAP_SIZE 1 << 16
#define FORKSRV_FD 198
#define FORK_WAIT_MULT 10
#define EXEC_TIMEOUT 1000
#define SHM_ENV_VAR "__AFL_SHM_ID"
#define SA struct sockaddr
#define PORT 1234

#define MEM_BARRIER() \
  __asm__ volatile("" ::: "memory")
#define likely(_x)   __builtin_expect(!!(_x), 1)
#define unlikely(_x)  __builtin_expect(!!(_x), 0)

enum {
  /* 00 */ FAULT_NONE,
  /* 01 */ FAULT_TMOUT,
  /* 02 */ FAULT_CRASH,
  /* 03 */ FAULT_ERROR,
  /* 04 */ FAULT_NOINST,
  /* 05 */ FAULT_NOBITS
};

static int forksrv_pid, fsrv_st_fd, fsrv_ctl_fd, shm_id, child_pid = -1; 
static int out_fd, dev_null_fd, child_timed_out, total_execs;
static uint8_t* trace_bits;
static uint8_t virgin_bits[MAP_SIZE];

static void fatal(char* msg) {
  fprintf(stderr, "%s", msg);
  exit(EXIT_FAILURE);
}

static void remove_shm(void) {
  shmctl(shm_id, IPC_RMID, NULL);
}

static void handle_timeout(int sig) {
  if (child_pid > 0) {
    child_timed_out = 1;
    kill(child_pid, SIGKILL);
  } else if (child_pid == -1 && forksrv_pid > 0) {
    child_timed_out = 1;
    kill(forksrv_pid, SIGKILL);
  }
}

static void setup_shm(void) {
  char shm_str[10];
  shm_id = shmget(IPC_PRIVATE, MAP_SIZE, IPC_CREAT | IPC_EXCL | 0600);
  if (shm_id < 0) fatal("shmget() failed");
  atexit(remove_shm);
  sprintf(shm_str, "%d", shm_id);
  setenv(SHM_ENV_VAR, shm_str, 1);
  memset(virgin_bits, 255, MAP_SIZE);
  trace_bits = shmat(shm_id, NULL, 0);
  if (trace_bits == (void *)-1) fatal("shmat() failed");
}

static void setup_signal_handlers() {
  struct sigaction sa;

  sa.sa_handler   = NULL;
  sa.sa_flags     = SA_RESTART;
  sa.sa_sigaction = NULL;

  sigemptyset(&sa.sa_mask);

  sa.sa_handler = handle_timeout;
  sigaction(SIGALRM, &sa, NULL);
}

static void init_forkserver(char* target_path) {
  static struct itimerval it;
  int st_pipe[2], ctl_pipe[2];
  int status, rlen;

  printf("[+] Spinning up the fork server...\n");
  if (pipe(st_pipe) || pipe(ctl_pipe)) fatal("pipe() failed");

  forksrv_pid = fork();
  if (forksrv_pid < 0) fatal("fork() failed");

  if (!forksrv_pid) {
    setsid();

    dup2(dev_null_fd, 1);
    dup2(dev_null_fd, 2);

    dup2(out_fd, 0);
    close(out_fd);

    if (dup2(ctl_pipe[0], FORKSRV_FD) < 0) fatal("dup2() failed");
    if (dup2(st_pipe[1], FORKSRV_FD + 1) < 0) fatal("dup2() failed");

    close(ctl_pipe[0]);
    close(ctl_pipe[1]);
    close(st_pipe[0]);
    close(st_pipe[1]);

    char** argv = { NULL };
    execv(target_path, argv);
    exit(EXIT_SUCCESS);
  }

  close(ctl_pipe[0]);
  close(st_pipe[1]);

  fsrv_ctl_fd = ctl_pipe[1];
  fsrv_st_fd = st_pipe[0]; 

  it.it_value.tv_sec = ((EXEC_TIMEOUT * FORK_WAIT_MULT) / 1000);
  it.it_value.tv_usec = ((EXEC_TIMEOUT * FORK_WAIT_MULT) % 1000) * 1000;

  setitimer(ITIMER_REAL, &it, NULL);

  rlen = read(fsrv_st_fd, &status, 4);

  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;

  setitimer(ITIMER_REAL, &it, NULL);

  if (rlen == 4) {
    printf("[+] All right - fork server is up.\n");
    return;
  }

  if (waitpid(forksrv_pid, &status, 0) <= 0) fatal("waitpid() failed");
  fatal("[+] Fork server handshake failed");
}

static uint8_t has_new_bits(uint8_t* virgin_map) {
  uint64_t* current = (uint64_t*)trace_bits;
  uint64_t* virgin  = (uint64_t*)virgin_map;
  int i = (MAP_SIZE >> 3);
  uint8_t ret = 0;

  while (i--) {
    if (unlikely(*current) && unlikely(*current & *virgin)) {
      if (likely(ret < 2)) {
        uint8_t* cur = (uint8_t*)current;
        uint8_t* vir = (uint8_t*)virgin;

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

static int run_target(char* fname, int len) {
  static struct itimerval it;
  static int prev_timed_out = 0;
  int status, res, kill_signal;
  char use_mem[len];

  int fd = open(fname, O_RDONLY);
  if (fd < 0) fatal("open() failed");
  if (read(fd, use_mem, len) != len) fatal("read() failed");
  close(fd);
  lseek(out_fd, 0, SEEK_SET);
  if (write(out_fd, use_mem, len) != len) fatal("write() failed");
  if (ftruncate(out_fd, len)) fatal("ftruncate() failed");
  lseek(out_fd, 0, SEEK_SET);

  memset(trace_bits, 0, MAP_SIZE);
  MEM_BARRIER();
  child_timed_out = 0;

  if ((res = write(fsrv_ctl_fd, &prev_timed_out, 4)) != 4) {
    fatal("Unable to request new process from fork server (OOM?)");
  } 
  if ((res = read(fsrv_st_fd, &child_pid, 4)) != 4) {
    fatal("Unable to request new process from fork server (OOM?)");
  } 
  if (child_pid <= 0) fatal("Fork server is misbehaving (OOM?)");

  it.it_value.tv_sec = (EXEC_TIMEOUT / 1000);
  it.it_value.tv_usec = (EXEC_TIMEOUT % 1000) * 1000;
  setitimer(ITIMER_REAL, &it, NULL);

  if ((res = read(fsrv_st_fd, &status, 4)) != 4) { // get return status
    fatal("Unable to communicate with fork server (OOM?)");
  }

  if (!WIFSTOPPED(status)) child_pid = 0;

  getitimer(ITIMER_REAL, &it);
  it.it_value.tv_sec = 0;
  it.it_value.tv_usec = 0;
  setitimer(ITIMER_REAL, &it, NULL);

  prev_timed_out = child_timed_out;
  total_execs ++;
  MEM_BARRIER();

  if (WIFSIGNALED(status)) {
    kill_signal = WTERMSIG(status);
    if (child_timed_out && kill_signal == SIGKILL) return FAULT_TMOUT;
    return FAULT_CRASH;
  }

  return FAULT_NONE;
}

// static void fuzz_all(char* dir) {
  // struct dirent **nl;
  // int nl_cnt;
  // nl_cnt = scandir(dir, &nl, NULL, alphasort);
  // if (nl_cnt < 0) fatal("scandir() failed");
  // for (int i = 0; i < nl_cnt; i++) {
    // struct stat st;
    // char fn[255];
    // sprintf(fn, "%s/%s", dir, nl[i]->d_name);
    // free(nl[i]);
    // if (lstat(fn, &st) || access(fn, R_OK)) fatal("access failed()");
    // if (!S_ISREG(st.st_mode) || !st.st_size || strstr(fn, "/README.txt")) continue;
    // Fuzz and find
    // int ret = run_target(fn, st.st_size);
    // int hnb = has_new_bits(virgin_bits);
    // printf("%s:%d:%d\n", fn, ret, hnb);
  // }
//
  // free(nl);
// }
static void server_up(void) {
  struct sockaddr_in server, client;
  int rlen, slen, server_fd, client_fd, flag = 1;

  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) fatal("socket() failed");

  server.sin_family = AF_INET;
	server.sin_addr.s_addr = INADDR_ANY;
	server.sin_port = htons(PORT);

  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));

  if (bind(server_fd,(SA*)&server, sizeof(server)) < 0) fatal("bind() failed");
  if (listen(server_fd, 3) < 0) fatal("listen() failed");
  printf("[+] waiting for incoming connections\n");

  socklen_t tmp = sizeof(client);
  client_fd = accept(server_fd, (SA*)&client, &tmp);
  if (client_fd < 0) fatal("accept() failed");
  
  char buf[1024];
  rlen = recv(client_fd, buf, sizeof(buf), 0);
  if (rlen < 0) fatal("recv() failed");
  slen = send(client_fd, buf, rlen, 0);
  if (slen < 0) fatal("send() failed");

  printf("[+] close and exit\n");

  close(client_fd);
  close(server_fd);
}

static void usage(char* name) {
  printf("Usage: %s /path/to/fuzzed_app\n", name);
  exit(EXIT_SUCCESS);
}

int main(int argv, char** argc) {

  if (argv < 2) usage(argc[0]);
  char* target_path = argc[1];

  unlink(".cur_input");
  out_fd = open(".cur_input", O_RDWR | O_CREAT | O_EXCL, 0600);
  if (out_fd < 0) fatal("Unable to create '.cur_input'");

  dev_null_fd = open("/dev/null", O_RDWR);
  if (dev_null_fd < 0) fatal("Unable to open /dev/null");

  // setup_shm();
  // setup_signal_handlers();
  // init_forkserver(target_path);
  server_up();

  return 0;
}
